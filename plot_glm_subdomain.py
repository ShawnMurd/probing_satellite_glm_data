#!/usr/bin/env python3
"""
plot_glm_subdomain.py
==========================

Plot the subdomain bounding box defined in ``download_glm_subdomain.py``
on a Cartopy map with underlying topography, rivers, county borders, and
state borders.

The map uses Natural Earth features:
    * rivers        - 10m physical 'rivers_lake_centerlines'
    * county borders - 10m cultural 'admin_2_counties'
    * state borders - 10m cultural 'admin_1_states_provinces_lines'

Topography is real elevation data (not a road/label map) from NOAA's ETOPO
2022 global relief model, fetched via OPeNDAP for just the plotted bbox and
rendered with matplotlib's 'terrain' colormap. Use --no-topo to fall back to
a flat land/ocean background if the remote dataset is unavailable.

Natural Earth shapefiles are downloaded automatically by Cartopy on first
use (internet required once; they are cached afterwards).

Examples
--------
# Save to the default PNG and open an interactive window
python plot_glm_subdomain.py

# Save to a specific file, no interactive window
python plot_glm_subdomain.py --out subdomain.png --no-show

Requires: cartopy, matplotlib, numpy, xarray, netCDF4
    conda install -c conda-forge cartopy matplotlib numpy xarray netcdf4
"""

from __future__ import annotations

import argparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from download_glm_subdomain import DOWNLOAD_SUBDOMAIN

# Padding (degrees) added around the domain so the box doesn't touch the
# map edges.
PAD = 1.0

# The subdomain bounding box defined in download_glm_subdomain.py
SUBDOMAIN = DOWNLOAD_SUBDOMAIN

# NOAA ETOPO 2022 global relief model (60 arc-second), served via OPeNDAP.
# No login required; only the requested bbox is transferred.
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
    "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
)


def fetch_elevation(bbox: dict, pad: float = PAD):
    """Return (lon, lat, elev_m) 1D/1D/2D arrays of ETOPO 2022 elevation
    covering the padded bbox. Raises on network/dataset failure."""
    ds = xr.open_dataset(ETOPO_URL, engine="netcdf4")
    sub = ds["z"].sel(
        lat=slice(bbox["lat_min"] - pad, bbox["lat_max"] + pad),
        lon=slice(bbox["lon_min"] - pad, bbox["lon_max"] + pad),
    ).load()
    return sub["lon"].values, sub["lat"].values, sub.values


def build_map(ax, bbox: dict, topo: bool = True) -> None:
    """Draw base map + topography, rivers, counties, states. Data stays in
    PlateCarree; the axes projection (set by the caller) may differ."""
    ax.set_extent(
        [
            bbox["lon_min"] - PAD,
            bbox["lon_max"] + PAD,
            bbox["lat_min"] - PAD,
            bbox["lat_max"] + PAD,
        ],
        crs=ccrs.PlateCarree(),
    )

    mesh = None
    if topo:
        try:
            lon, lat, elev = fetch_elevation(bbox)
            # Reserve the colormap's blue segment for below-sea-level only.
            vmin = min(0.0, float(np.nanmin(elev)))
            vmax = float(np.nanmax(elev))
            mesh = ax.pcolormesh(
                lon, lat, elev,
                transform=ccrs.PlateCarree(),
                cmap="terrain", norm=Normalize(vmin=vmin, vmax=vmax),
                shading="nearest", zorder=0,
            )
        except Exception as exc:  # network/dataset unavailable
            print(f"Topography unavailable, using flat background ({exc})")
            topo = False

    if not topo:
        # Subtle land/ocean background
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f5f2e8", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dcecf5", zorder=0)

    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#dcecf5",
                   edgecolor="#7fa8c9", linewidth=0.4, zorder=1)

    # Rivers (10m detail for a regional-scale map)
    rivers = cfeature.NaturalEarthFeature(
        category="physical",
        name="rivers_lake_centerlines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(rivers, edgecolor="#4a90d9", linewidth=0.6, zorder=2)

    # County borders (thin, light) and state borders (thick, dark)
    counties = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_2_counties",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(counties, edgecolor="#b0b0b0", linewidth=0.4, zorder=3)

    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(states, edgecolor="#333333", linewidth=1.1, zorder=4)

    # National borders for context (domain straddles CO only, but cheap)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#333333",
                   linewidth=1.1, zorder=4)

    # Gridlines with lat/lon labels
    #gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#888888",
    #                  linestyle="--", zorder=5)
    #gl.top_labels = False
    #gl.right_labels = False

    return mesh


def draw_domain_box(ax, bbox: dict) -> None:
    """Overlay the subdomain bounding box and annotate its corners."""
    width = bbox["lon_max"] - bbox["lon_min"]
    height = bbox["lat_max"] - bbox["lat_min"]

    # Shaded box containing the subdomain (commented out)
    #rect = Rectangle(
    #    (bbox["lon_min"], bbox["lat_min"]),
    #    width,
    #    height,
    #    transform=ccrs.PlateCarree(),
    #    facecolor="red",
    #    alpha=0.2,
    #    linewidth=0,
    #    zorder=6,
    #)
    #ax.add_patch(rect)

    rect2 = Rectangle(
            (bbox["lon_min"], bbox["lat_min"]),
            width,
            height,
            transform=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="red",
            linewidth=3.0,
            zorder=6,
        )
    ax.add_patch(rect2)

    # Label the box
    #ax.text(
    #    bbox["lon_min"] + width / 2,
    #    bbox["lat_max"] + 0.08,
    #    "SUBDOMAIN",
    #    transform=ccrs.PlateCarree(),
    #    ha="center",
    #    va="bottom",
    #    fontsize=11,
    #    fontweight="bold",
    #    color="red",
    #    zorder=7,
    #)

    # Corner coordinate annotations
    #corners = [
    #    (bbox["lon_min"], bbox["lat_max"], "right", "bottom"),
    #    (bbox["lon_max"], bbox["lat_max"], "left", "bottom"),
    #    (bbox["lon_min"], bbox["lat_min"], "right", "top"),
    #    (bbox["lon_max"], bbox["lat_min"], "left", "top"),
    #]
    #for lon, lat, ha, va in corners:
    #    ax.plot(lon, lat, marker="o", markersize=4, color="red",
    #            transform=ccrs.PlateCarree(), zorder=7)
    #    ax.text(
    #        lon, lat, f" {lat:.1f}, {lon:.1f} ",
    #        transform=ccrs.PlateCarree(),
    #        ha=ha, va=va, fontsize=7, color="red", zorder=7,
    #    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot the subdomain GLM domain with Cartopy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", default="glm_subdomain.png",
                   help="Output image file (PNG/PDF/etc. by extension)")
    p.add_argument("--no_save", action="store_true",
                   help="Save the output image to the specified file")
    p.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open an interactive plot window")
    p.add_argument("--no-topo", action="store_true",
                   help="Skip ETOPO elevation data (flat land/ocean background)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bbox = SUBDOMAIN

    central_lon = (bbox["lon_min"] + bbox["lon_max"]) / 2
    central_lat = (bbox["lat_min"] + bbox["lat_max"]) / 2
    proj = ccrs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
        standard_parallels=(bbox["lat_min"], bbox["lat_max"]),
    )
    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"projection": proj}
    )

    mesh = build_map(ax, bbox, topo=not args.no_topo)
    draw_domain_box(ax, bbox)

    if mesh is not None:
        fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7,
                     pad=0.05, label="Elevation (m)")

    ax.set_title(
        "GLM Subdomain Domain\n"
        f"lat {bbox['lat_min']:.2f}..{bbox['lat_max']:.2f}, "
        f"lon {bbox['lon_min']:.2f}..{bbox['lon_max']:.2f}",
        fontsize=12,
    )

    if not args.no_save:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {args.out}")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
