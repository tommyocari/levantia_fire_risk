import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from pathlib import Path

DATA    = Path(__file__).parent.parent / "data"
OUT     = DATA / "fire_events_enriched.csv"

LAND_USE_LABELS = {1: "urban", 2: "forest", 3: "shrubland", 4: "agricultural", 5: "water"}
climate_vars = ["temperature", "humidity", "wind_speed", "precipitation", "ndvi"]
nc_path      = DATA / "climate_grid.nc"

print("Loading fire events...")
df = pd.read_csv(DATA / "fire_events.csv", parse_dates=["date"])
print(f"  {len(df)} events, date range: {df['date'].min().date()} -> {df['date'].max().date()}")

print("Loading climate grid...")
climate = xr.open_dataset(nc_path)[climate_vars].load()
print(f"  Grid: {dict(climate.dims)}")

# --- 1. Climate join: values on ignition day ---
print("Joining climate values on ignition day...")
points = climate.sel(
    time=xr.DataArray(df["date"].values, dims="points"),
    lat=xr.DataArray(df["lat"].values,   dims="points"),
    lon=xr.DataArray(df["lon"].values,   dims="points"),
    method="nearest"
)
for v in climate_vars:
    df[v] = points[v].values
print(f"  Done -- added {climate_vars}")

# --- 2. Fire weather window: 7-day rolling mean ending on ignition day ---
print("Computing 7-day rolling mean...")
climate_7d = climate.rolling(time=7, min_periods=1).mean()
points_7d  = climate_7d.sel(
    time=xr.DataArray(df["date"].values, dims="points"),
    lat=xr.DataArray(df["lat"].values,   dims="points"),
    lon=xr.DataArray(df["lon"].values,   dims="points"),
    method="nearest"
)
for v in climate_vars:
    df[f"{v}_7d"] = points_7d[v].values
print(f"  Done -- added {[f'{v}_7d' for v in climate_vars]}")

# --- 3. Fire Weather Index (FWI) ---
# Simplified proxy: drought factor x wind amplification x precipitation suppression
print("Computing Fire Weather Index...")
def fwi(T, H, W, P):
    return (T / 30) * ((100 - H) / 100) * (1 + W / 20) * np.exp(-P / 10)

df["fwi"]    = fwi(df["temperature"],    df["humidity"],    df["wind_speed"],    df["precipitation"]).clip(lower=0)
df["fwi_7d"] = fwi(df["temperature_7d"], df["humidity_7d"], df["wind_speed_7d"], df["precipitation_7d"]).clip(lower=0)
print(f"  Done -- fwi range: [{df['fwi'].min():.2f}, {df['fwi'].max():.2f}]")

# --- 4. Anomalies relative to 10-year climatological baseline ---
# For each variable: anomaly = day-of value minus the long-term monthly mean
print("Computing climatological anomalies...")
climatology = climate.groupby("time.month").mean("time")  # (month=12, lat=50, lon=50)
months      = xr.DataArray(df["date"].dt.month.values, dims="points")
clim_points = climatology.sel(
    month=months,
    lat=xr.DataArray(df["lat"].values, dims="points"),
    lon=xr.DataArray(df["lon"].values, dims="points"),
    method="nearest"
)
for v in climate_vars:
    df[f"{v}_anom"] = df[v] - clim_points[v].values
print(f"  Done -- added {[f'{v}_anom' for v in climate_vars]}")

climate.close()

# --- 5. Land use join ---
print("Joining land use...")
with rasterio.open(DATA / "land_use.tif") as src:
    lu        = src.read(1)
    transform = src.transform

rows, cols = rasterio.transform.rowcol(transform, df["lon"].values, df["lat"].values)
rows = np.clip(rows, 0, lu.shape[0] - 1)
cols = np.clip(cols, 0, lu.shape[1] - 1)
df["land_use"] = [LAND_USE_LABELS.get(int(c), "unknown") for c in lu[rows, cols]]
print(f"  Done -- land use distribution: {df['land_use'].value_counts().to_dict()}")

print("\nWriting output...")
df.to_csv(OUT, index=False)
print(f"  Written {len(df)} rows -> {OUT}")
print(f"  Columns: {list(df.columns)}")
