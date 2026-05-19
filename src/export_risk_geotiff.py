import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from risk_map import load_static_data, compute_prob_grid, DATA

DATE = "2020-07-15"  # summer day to export

# --------------------------------------------------------------------------- #
# Compute risk grid
# --------------------------------------------------------------------------- #
print("Loading static data...")
static = load_static_data()

print(f"Computing risk map for {DATE}...")
prob_grid = compute_prob_grid(DATE, static)
static["climate_grid"].close()

lats = static["lats"]   # (50,) ascending: 36 → 46
lons = static["lons"]   # (50,) ascending:  6 → 18
NY, NX = static["NY"], static["NX"]

# --------------------------------------------------------------------------- #
# Write GeoTIFF
# GeoTIFF convention: row 0 = northernmost → flip the grid vertically
# from_bounds: (west, south, east, north, width, height)
# --------------------------------------------------------------------------- #
transform = from_bounds(
    west=lons.min(), south=lats.min(),
    east=lons.max(), north=lats.max(),
    width=NX, height=NY,
)

out_path = DATA.parent / "notebooks" / f"risk_map_{DATE}.tif"

with rasterio.open(
    out_path, "w",
    driver="GTiff",
    height=NY, width=NX,
    count=1,
    dtype="float32",
    crs=CRS.from_epsg(4326),
    transform=transform,
) as dst:
    dst.write(np.flipud(prob_grid).astype("float32"), 1)

print(f"Saved to {out_path}")
print("Open in QGIS: Layer > Add Layer > Add Raster Layer")
