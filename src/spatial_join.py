import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from pathlib import Path

DATA    = Path(__file__).parent.parent / "data"
OUT     = DATA / "fire_events_enriched.csv"

LAND_USE_LABELS = {1: "urban", 2: "forest", 3: "shrubland", 4: "agricultural", 5: "water"}

# --- Load inputs ---
df      = pd.read_csv(DATA / "fire_events.csv", parse_dates=["date"])
climate_vars = ["temperature", "humidity", "wind_speed", "precipitation", "ndvi"]
nc_path = DATA / "climate_grid.nc"
climate = xr.open_dataset(nc_path)[climate_vars].load()
points = climate.sel(
    time=xr.DataArray(df["date"].values, dims="points"),
    lat=xr.DataArray(df["lat"].values, dims="points"),
    lon=xr.DataArray(df["lon"].values, dims="points"),
    method="nearest"
)
for v in climate_vars:
    df[v] = points[v].values

climate.close()

# --- 2. Land use join ---
with rasterio.open(DATA / "land_use.tif") as src:
    lu        = src.read(1)
    transform = src.transform

rows, cols = rasterio.transform.rowcol(transform, df["lon"].values, df["lat"].values)
rows = np.clip(rows, 0, lu.shape[0] - 1) # in case of lat, lon bigger than expected one 
cols = np.clip(cols, 0, lu.shape[1] - 1)
df["land_use"] = [LAND_USE_LABELS.get(int(c), "unknown") for c in lu[rows, cols]]

df.to_csv(OUT, index=False)
print(f"Written {len(df)} rows → {OUT}")
print(f"Columns: {list(df.columns)}")
