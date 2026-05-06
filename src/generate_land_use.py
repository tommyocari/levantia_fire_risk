import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path

SEED = 42
NX, NY = 50, 50
LAT_MIN, LAT_MAX = 36.0, 46.0
LON_MIN, LON_MAX = 6.0, 18.0
OUT_PATH = Path(__file__).parent.parent / "data" / "land_use.tif"

# class encoding
CLASSES = {
    "urban":        1,
    "forest":       2,
    "shrubland":    3,
    "agricultural": 4,
    "water":        5,
}

# target proportions (Mediterranean-like)
PROPORTIONS = {
    "urban":        0.05,
    "forest":       0.30,
    "shrubland":    0.30,
    "agricultural": 0.25,
    "water":        0.10,
}

rng = np.random.default_rng(SEED)

# place random Voronoi seed points, one class per seed (weighted by proportion)
N_SEEDS = 60
seed_y = rng.integers(0, NY, N_SEEDS)
seed_x = rng.integers(0, NX, N_SEEDS)
class_names = list(CLASSES.keys())
seed_classes = rng.choice(class_names, size=N_SEEDS, p=list(PROPORTIONS.values()))

# for each grid cell, assign the class of the nearest seed
yy, xx = np.mgrid[0:NY, 0:NX]                                    # (NY, NX) index grids
dy = yy[:, :, None] - seed_y[None, None, :]                      # (NY, NX, N_SEEDS)
dx = xx[:, :, None] - seed_x[None, None, :]
dist = dy**2 + dx**2                                              # squared distance
nearest = np.argmin(dist, axis=2)                                 # (NY, NX)

vfunc = np.vectorize(lambda i: CLASSES[seed_classes[i]])
land_use = vfunc(nearest).astype(np.uint8)

# rasterio transform: maps pixel indices to geographic coordinates
transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, NX, NY)
crs = CRS.from_epsg(4326)  # WGS84

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(
    OUT_PATH, "w",
    driver="GTiff",
    height=NY, width=NX,
    count=1,
    dtype=np.uint8,
    crs=crs,
    transform=transform,
    nodata=0,
) as dst:
    dst.write(land_use, 1)
    dst.update_tags(
        classes="1=urban 2=forest 3=shrubland 4=agricultural 5=water",
        source="generate_land_use.py",
    )

# summary
names = list(CLASSES.keys())
print(f"Written land_use.tif → {OUT_PATH}")
for name, code in CLASSES.items():
    pct = (land_use == code).mean() * 100
    print(f"  {name:15s} (code {code}): {pct:.1f}%")
