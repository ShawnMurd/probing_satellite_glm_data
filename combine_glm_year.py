#!/usr/bin/env python3
"""
combine_glm_year.py
====================

Combine all daily GLM subset files (``<outdir>/YYYY/DDD.nc``,
as produced by ``download_glm_subdomain.py``) for one year into a single
netCDF file: ``<outdir>/YYYY.nc``.

If the destination file already exists, it is deleted before the new one
is written.

Usage
-----
python combine_glm_year.py 2022
python combine_glm_year.py 2022 --outdir glm_data
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import xarray as xr

GB = 1e9  # bytes per "GB"


def combine_year(outdir: Path, year: int) -> Path:
    """Combine all daily subset files for ``year`` into ``<outdir>/YYYY.nc``."""
    year_dir = outdir / f"{year}"
    files = sorted(year_dir.glob("*.nc"))
    if not files:
        raise SystemExit(f"No daily files found in {year_dir}")

    dst = outdir / f"{year}.nc"
    if dst.exists():
        dst.unlink()

    # .load() closes the source file handle immediately, avoiding lazy
    # reopens during concat/to_netcdf that can leave handles open.
    datasets = []
    for f in files:
        with xr.open_dataset(f, engine="netcdf4", decode_times=False) as d:
            datasets.append(d.load())

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
        f"{dt.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ} combined {len(files)} daily "
        "subsets by combine_glm_year.py"
    )

    combined.to_netcdf(dst, engine="netcdf4", encoding=encoding)
    combined.close()
    return dst


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Combine daily GLM subsets for one year into a single netCDF file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("year", type=int, help="Year to combine, e.g. 2022")
    p.add_argument("--outdir", default="glm_data",
                   help="Directory containing <year>/ subfolders of daily .nc files")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    dst = combine_year(outdir, args.year)
    print(f"Combined -> {dst} ({dst.stat().st_size / GB:.3f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
