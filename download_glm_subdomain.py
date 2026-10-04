#!/usr/bin/env python3
"""
download_glm_subdomain.py
==========================

Download GOES Geostationary Lightning Mapper (GLM) Level-2 LCFA data from
NOAA's public AWS S3 archive and keep only lightning over the desired subdomain.

How it works
------------
1. Lists GLM-L2-LCFA files on S3 for the requested date/hour window.
2. Downloads files one at a time (stops before exceeding ``--max-gb``).
3. Subsets each file to flash-level data over the desired subdomain bounding box:
     * group and event data are dropped entirely (flash-level only);
     * only flashes with ``flash_quality_flag == 0`` whose centroid falls
       inside the box are kept.
4. Writes the subset as a compressed netCDF file and deletes the raw file
   (unless ``--keep-raw``), so the retained dataset is typically only a
   small fraction of the downloaded volume.
5. At the end of each day (once all its hourly subsets are written), the
   day's subset files are combined into a single ``<outdir>/YYYY/DDD.nc``
   file and the individual hourly files (and now-empty directories) are
   deleted.

Examples
--------
# Estimate sizes only (no downloads)
python download_glm_subdomain.py --start 2024-06-01 --end 2024-06-30 --dry-run

# June 2024, GOES-16, all hours, stop at 100 GB downloaded
python download_glm_subdomain.py --start 2024-06-01 --end 2024-06-30

# Afternoon/evening convection only (18-23 UTC), GOES-19
python download_glm_subdomain.py --start 2025-06-01 --end 2025-06-30 \
    --satellite 19 --hours 18-23

Notes
-----
* Satellite choice: GOES-16 (GOES-East) through early 2025; GOES-19 is
  GOES-East from ~April 2025 onward. GOES-18 (GOES-West) also covers
  Colorado well. Both East and West view the Front Range.
* Requires: boto3, numpy, xarray, netCDF4
      conda install -c conda-forge boto3 numpy xarray netcdf4
* Re-running the script skips subsets that already exist (resumable).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import tempfile
from pathlib import Path

import boto3
import numpy as np
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUCKETS = {
    16: "noaa-goes16",
    17: "noaa-goes17",
    18: "noaa-goes18",
    19: "noaa-goes19",
}
PRODUCT = "GLM-L2-LCFA"  # Lightning Cluster-Filter Algorithm (events/groups/flashes)

# Colorado Front Range corridor
#DOWNLOAD_SUBDOMAIN = {
#    "lat_min": 38.6,
#    "lat_max": 41.0,
#    "lon_min": -106.13,
#    "lon_max": -104.8,
#}

# All of Western Colorado
DOWNLOAD_SUBDOMAIN = {
    "lat_min": 37,
    "lat_max": 41.0,
    "lon_min": -109.05,
    "lon_max": -104.8,
}

GB = 1e9  # bytes per "GB" for the download cap


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def make_client():
    """Anonymous (unsigned) client for the public NOAA GOES buckets."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def iter_prefixes(start: dt.date, end: dt.date, hours):
    """Yield 'YYYY/DDD/HH' S3 prefixes for each day/hour in the window."""
    day = start
    while day <= end:
        doy = day.timetuple().tm_yday
        for hour in hours:
            yield f"{day.year}/{doy:03d}/{hour:02d}"
        day += dt.timedelta(days=1)


def list_objects(client, bucket: str, prefix: str):
    """Yield (key, size_bytes) for every GLM file under one hour prefix."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{PRODUCT}/{prefix}/"):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


# ---------------------------------------------------------------------------
# Subsetting
# ---------------------------------------------------------------------------
def _rebase_time_offsets(ds: xr.Dataset) -> None:
    """Rebase per-flash time-offset variables onto the fixed product_time epoch, in place.

    Each raw granule's ``flash*_time_offset_of_*`` variables are encoded as
    seconds since *that granule's own* ~20 s start time. Concatenating many
    granules (e.g. across a whole day) without correcting for this leaves
    every flash's offset clustered within one scan window instead of
    spanning the true time range, since only the first granule's "seconds
    since ..." reference survives the combine. Adding this granule's
    ``product_time`` (itself seconds since the fixed 2000-01-01 12:00:00
    epoch) converts the offsets to a common absolute reference that stays
    correct after concatenation.
    """
    epoch_units = ds["product_time"].attrs["units"]
    product_time = ds["product_time"].values
    for name, var in ds.variables.items():
        units = var.attrs.get("units", "")
        if name == "product_time" or not units.startswith("seconds since") or units == epoch_units:
            continue
        var.values = var.values + product_time
        var.attrs["units"] = epoch_units


def subset_to_bbox(src_path: str, dst_path: Path, bbox: dict) -> bool:
    """Subset one GLM LCFA file to flash-level data over the bounding box.

    Group and event data are dropped entirely (flash-level only). Only
    flashes with ``flash_quality_flag == 0`` whose centroid falls inside
    the box are kept.

    Returns True if any flashes were kept (subset written to ``dst_path``),
    False if the file contains no qualifying lightning inside the box.
    """
    # decode_times=False keeps original packed integer encodings intact,
    # which makes the re-write faithful and compact.
    with xr.open_dataset(src_path, engine="netcdf4", decode_times=False) as ds:
        # Drop all group/event variables; keep flash-level + scalar/metadata ones.
        keep_vars = [
            name for name, var in ds.variables.items()
            if "number_of_groups" not in var.dims and "number_of_events" not in var.dims
        ]
        ds = ds[keep_vars]

        in_box = (
            (ds.flash_lat >= bbox["lat_min"])
            & (ds.flash_lat <= bbox["lat_max"])
            & (ds.flash_lon >= bbox["lon_min"])
            & (ds.flash_lon <= bbox["lon_max"])
        )
        good_quality = ds.flash_quality_flag == 0

        f_idx = np.flatnonzero((in_box & good_quality).values)
        if f_idx.size == 0:
            return False

        sub = ds.isel(number_of_flashes=f_idx)
        _rebase_time_offsets(sub)

        # Avoid the classic "_FillValue present in both attrs and encoding"
        # netCDF write conflict.
        for name in sub.variables:
            var = sub[name]
            if "_FillValue" in var.attrs and "_FillValue" in var.encoding:
                del var.attrs["_FillValue"]

        # Deflate all array variables to keep the subset small on disk.
        encoding = {
            name: {"zlib": True, "complevel": 4}
            for name, var in sub.data_vars.items()
            if var.ndim > 0
        }

        sub.attrs["subset_note"] = (
            f"Subset to subdomain box: lat {bbox['lat_min']}..{bbox['lat_max']}, "
            f"lon {bbox['lon_min']}..{bbox['lon_max']} (flash-centroid based, "
            "flash_quality_flag == 0 only, group/event data dropped)"
        )
        sub.attrs["history"] = (
            f"{dt.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ} subset by "
            "download_glm_subdomain.py"
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_netcdf(dst_path, engine="netcdf4", encoding=encoding)
    return True


# ---------------------------------------------------------------------------
# Daily combination
# ---------------------------------------------------------------------------
def combine_day(outdir: Path, year: int, doy: int) -> Path | None:
    """Combine all hourly subset files for one day into a single file.

    The individual files (and now-empty hour/day directories) that were
    combined are deleted afterwards. Returns the path to the combined file,
    or None if there were no files to combine.
    """
    day_dir = outdir / f"{year}" / f"{doy:03d}"
    files = sorted(day_dir.glob("*/*.nc"))
    if not files:
        return None

    # .load() pulls all data into memory and closes the underlying file
    # handle immediately; without it, xarray's lazy file manager can
    # silently reopen the source file during concat/to_netcdf, leaving a
    # handle open on Windows that blocks the unlink() below.
    datasets = []
    for f in files:
        with xr.open_dataset(f, engine="netcdf4", decode_times=False) as d:
            datasets.append(d.load())

    # minimal/override: only variables that already vary along
    # number_of_flashes get concatenated; scalar metadata (which may
    # differ slightly file-to-file) is taken from the first file.
    combined = xr.concat(
        datasets, dim="number_of_flashes",
        data_vars="minimal", coords="minimal", compat="override",
    )

    for name in combined.variables:
        var = combined[name]
        if "_FillValue" in var.attrs and "_FillValue" in var.encoding:
            del var.attrs["_FillValue"]

    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, var in combined.data_vars.items()
        if var.ndim > 0
    }
    combined.attrs["history"] = (
        f"{dt.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ} combined {len(files)} hourly "
        "subsets by download_glm_subdomain.py"
    )

    dst = outdir / f"{year}" / f"{doy:03d}.nc"
    combined.to_netcdf(dst, engine="netcdf4", encoding=encoding)
    combined.close()

    for f in files:
        f.unlink()
    for hour_dir in sorted(day_dir.iterdir()):
        if hour_dir.is_dir() and not any(hour_dir.iterdir()):
            hour_dir.rmdir()
    if day_dir.exists() and not any(day_dir.iterdir()):
        day_dir.rmdir()

    return dst


def _combine_and_report(outdir: Path, day_key: tuple[int, int]) -> None:
    year, doy = day_key
    dst = combine_day(outdir, year, doy)
    if dst is not None:
        print(f"Combined day {year}/{doy:03d} -> {dst} "
              f"({dst.stat().st_size / GB:.3f} GB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_hours(text: str | None):
    """Parse --hours: '18-23' (inclusive range) or '18,20,22'. None = all."""
    if text is None:
        return list(range(24))
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download GOES GLM L2 data and subset to the desired subdomain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", required=True, help="Start date, YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="End date, YYYY-MM-DD (inclusive)")
    p.add_argument("--satellite", type=int, choices=sorted(BUCKETS), default=16,
                   help="GOES satellite number (16=East<=2024, 19=East>=2025, 18=West)")
    p.add_argument("--hours", default=None,
                   help="UTC hours to include, e.g. '18-23' or '18,20,22'. Default: all")
    p.add_argument("--outdir", default="glm_data", help="Output directory")
    p.add_argument("--max-gb", type=float, default=100.0,
                   help="Stop before cumulative downloads exceed this many GB")
    p.add_argument("--keep-raw", action="store_true",
                   help="Also keep the full (unsubsetted) files under <outdir>/raw")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process subsets that already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="Only report the total size of matching files; download nothing")

    # Force user to change lat/lon coordinates of box in this script manually
    # b/c DOWNLOAD_SUBDOMAIN from this script is used by other scripts
    #p.add_argument("--lat-min", type=float, default=DOWNLOAD_SUBDOMAIN["lat_min"])
    #p.add_argument("--lat-max", type=float, default=DOWNLOAD_SUBDOMAIN["lat_max"])
    #p.add_argument("--lon-min", type=float, default=DOWNLOAD_SUBDOMAIN["lon_min"])
    #p.add_argument("--lon-max", type=float, default=DOWNLOAD_SUBDOMAIN["lon_max"])

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    hours = parse_hours(args.hours)
    bbox = DOWNLOAD_SUBDOMAIN
    bucket = BUCKETS[args.satellite]
    outdir = Path(args.outdir)
    cap_bytes = args.max_gb * GB

    client = make_client()

    print(f"Bucket   : s3://{bucket}/{PRODUCT}")
    print(f"Window   : {start} .. {end}, hours {hours[0]:02d}-{hours[-1]:02d} UTC")
    print(f"BBox     : lat {bbox['lat_min']}..{bbox['lat_max']}, "
          f"lon {bbox['lon_min']}..{bbox['lon_max']}")
    print(f"Cap      : {args.max_gb:g} GB downloaded")
    if args.dry_run:
        print("Mode     : DRY RUN (no downloads)\n")

    total_bytes = 0
    n_files = 0
    n_kept = 0
    n_empty = 0
    n_skipped_existing = 0
    kept_bytes = 0
    current_day = None  # (year, doy) of the day currently being processed

    for prefix in iter_prefixes(start, end, hours):
        year_str, doy_str, _hour_str = prefix.split("/")
        day_key = (int(year_str), int(doy_str))
        if not args.dry_run and current_day is not None and day_key != current_day:
            _combine_and_report(outdir, current_day)
        current_day = day_key

        for key, size in list_objects(client, bucket, prefix):
            n_files += 1

            if args.dry_run:
                total_bytes += size
                continue

            # Destination mirrors the S3 layout: <outdir>/YYYY/DDD/HH/<file>
            dst = outdir / Path(key).parent.relative_to(PRODUCT) / Path(key).name
            if dst.exists() and not args.overwrite:
                n_skipped_existing += 1
                continue

            # Enforce the download cap *before* fetching the file.
            if total_bytes + size > cap_bytes:
                print(f"\nReached download cap of {args.max_gb:g} GB "
                      f"({total_bytes / GB:.2f} GB used). Stopping.")
                _combine_and_report(outdir, current_day)
                _print_summary(n_files, n_kept, n_empty, n_skipped_existing,
                               total_bytes, kept_bytes)
                return 0

            # --- download (to temp unless --keep-raw) ---------------------
            if args.keep_raw:
                raw_path = outdir / "raw" / Path(key).parent.relative_to(PRODUCT) / Path(key).name
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(raw_path))
                src, tmp = str(raw_path), None
            else:
                fd, tmp = tempfile.mkstemp(suffix=".nc")
                os.close(fd)
                client.download_file(bucket, key, tmp)
                src = tmp
            total_bytes += size

            # --- subset ----------------------------------------------------
            try:
                kept = subset_to_bbox(src, dst, bbox)
            except Exception as exc:  # corrupt file, read error, etc.
                print(f"  ! failed {Path(key).name}: {exc}")
                kept = False
            finally:
                if tmp is not None:
                    Path(tmp).unlink(missing_ok=True)

            if kept:
                n_kept += 1
                kept_bytes += dst.stat().st_size
            else:
                n_empty += 1

            if n_kept % 25 == 0 and n_kept > 0:
                print(f"  ... {total_bytes / GB:6.2f} GB downloaded, "
                      f"{n_kept} files with subdomain lightning "
                      f"({kept_bytes / GB:.3f} GB kept)")

        print(f"hour {prefix}: running total {total_bytes / GB:.2f} GB, "
              f"{n_kept} kept, {n_empty} empty")

    if not args.dry_run and current_day is not None:
        _combine_and_report(outdir, current_day)

    _print_summary(n_files, n_kept, n_empty, n_skipped_existing,
                   total_bytes, kept_bytes, dry_run=args.dry_run)
    return 0


def _print_summary(n_files, n_kept, n_empty, n_skipped, total_bytes, kept_bytes,
                   dry_run=False):
    print("\n" + "=" * 60)
    if dry_run:
        print(f"Matching files          : {n_files}")
        print(f"Total download size     : {total_bytes / GB:.2f} GB")
        print(f"Estimated kept subset   : ~{total_bytes / GB * 0.03:.1f} GB "
              "(typical subset is ~2-5% of raw)")
    else:
        print(f"Files examined          : {n_files}")
        print(f"Already on disk (skip)  : {n_skipped}")
        print(f"Files with subdomain lightning : {n_kept}")
        print(f"Files empty over subdomain     : {n_empty}")
        print(f"Downloaded volume       : {total_bytes / GB:.2f} GB")
        print(f"Retained subset volume  : {kept_bytes / GB:.3f} GB")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
