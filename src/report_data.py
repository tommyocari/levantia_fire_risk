import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from pathlib import Path
from datetime import datetime

DATA    = Path(__file__).parent.parent / "data"
OUT     = DATA / "data_quality_report.txt"
lines   = []

def section(title):
    lines.append(f"\n{'='*50}")
    lines.append(f"  {title}")
    lines.append(f"{'='*50}")

def row(label, value):
    lines.append(f"  {label:<30} {value}")

lines.append(f"Data Quality Report — generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- climate_grid.nc ---
section("climate_grid.nc")
ds = xr.open_dataset(DATA / "climate_grid.nc")
row("Time range", f"{str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]}")
row("Time steps", str(len(ds.time)))
row("Grid",       f"{len(ds.lat)} lat × {len(ds.lon)} lon")
row("Lat bounds", f"[{float(ds.lat.min()):.2f}, {float(ds.lat.max()):.2f}]")
row("Lon bounds", f"[{float(ds.lon.min()):.2f}, {float(ds.lon.max()):.2f}]")
row("Variables",  ", ".join(ds.data_vars))
for v in ds.data_vars:
    mn  = float(ds[v].min())
    mx  = float(ds[v].max())
    nan = int(ds[v].isnull().sum())
    row(f"  {v}", f"[{mn:.2f}, {mx:.2f}]  missing: {nan}")
ds.close()

# --- fire_events.csv ---
section("fire_events.csv")
df = pd.read_csv(DATA / "fire_events.csv", parse_dates=["date"])
row("Rows", str(len(df)))
row("Date range",  f"{df.date.min().date()} → {df.date.max().date()}")
row("Lat bounds",  f"[{df.lat.min():.2f}, {df.lat.max():.2f}]")
row("Lon bounds",  f"[{df.lon.min():.2f}, {df.lon.max():.2f}]")
row("Missing values", str(df.isnull().sum().sum()))
row("burned_area_ha", f"[{df.burned_area_ha.min():.1f}, {df.burned_area_ha.max():.1f}]  median: {df.burned_area_ha.median():.1f}")
row("duration_days",  f"[{df.duration_days.min()}, {df.duration_days.max()}]  median: {df.duration_days.median():.0f}")
row("ignition_cause", df.ignition_cause.value_counts().to_dict().__str__())

# --- land_use.tif ---
section("land_use.tif")
with rasterio.open(DATA / "land_use.tif") as src:
    lu     = src.read(1)
    bounds = src.bounds
    crs    = src.crs
row("Shape",   f"{lu.shape[0]} × {lu.shape[1]}")
row("CRS",     str(crs.to_epsg()))
row("Bounds",  f"lon [{bounds.left:.1f}, {bounds.right:.1f}]  lat [{bounds.bottom:.1f}, {bounds.top:.1f}]")
row("Missing", str(int(np.sum(lu == 0))))
codes, counts = np.unique(lu, return_counts=True)
for code, count in zip(codes, counts):
    row(f"  class {code}", f"{count} cells ({count/lu.size*100:.1f}%)")

# --- socioeconomic.csv ---
section("socioeconomic.csv")
df_s = pd.read_csv(DATA / "socioeconomic.csv")
row("Rows",           str(len(df_s)))
row("Missing values", str(df_s.isnull().sum().sum()))
for col in ["population", "gdp_per_capita", "infrastructure_density"]:
    row(f"  {col}", f"[{df_s[col].min():.1f}, {df_s[col].max():.1f}]  median: {df_s[col].median():.1f}")

lines.append("")
report = "\n".join(lines)
print(report)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(report)
print(f"\nReport saved → {OUT}")
