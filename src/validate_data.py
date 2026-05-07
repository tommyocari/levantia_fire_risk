import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
PASS, FAIL = "  OK ", " FAIL"

errors = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        errors.append(label)

print("\n=== climate_grid.nc ===")
ds = xr.open_dataset(DATA / "climate_grid.nc")

check("dimensions",       set(ds.dims) == {"time", "lat", "lon"})
check("variables",        set(ds.data_vars) >= {"temperature", "humidity", "wind_speed", "precipitation", "ndvi"})
check("time range",       str(ds.time.values[0])[:10] == "2014-01-01" and str(ds.time.values[-1])[:10] == "2023-12-31")
check("no missing values",all(ds[v].isnull().sum().item() == 0 for v in ds.data_vars))
check("dtypes float32",   all(ds[v].dtype == np.float32 for v in ds.data_vars))
check("temperature range",float(ds.temperature.min()) > -30 and float(ds.temperature.max()) < 60)
check("humidity range",   float(ds.humidity.min()) >= 0 and float(ds.humidity.max()) <= 100)
check("ndvi range",       float(ds.ndvi.min()) >= -1 and float(ds.ndvi.max()) <= 1)
check("precip non-negative", float(ds.precipitation.min()) >= 0)
check("wind non-negative",   float(ds.wind_speed.min()) >= 0)

climate_lat = ds.lat.values
climate_lon = ds.lon.values
ds.close()

print("\n=== fire_events.csv ===")
df = pd.read_csv(DATA / "fire_events.csv", parse_dates=["date"])

check("row count > 0",    len(df) > 0, f"{len(df)} rows")
check("columns present",  {"date","lat","lon","burned_area_ha","duration_days","ignition_cause"}.issubset(df.columns))
check("no missing values",df.isnull().sum().sum() == 0)
check("dtypes date",      pd.api.types.is_datetime64_any_dtype(df["date"]))
check("dtypes numeric",   df[["lat","lon","burned_area_ha","duration_days"]].select_dtypes(exclude='number').empty)
check("burned_area > 0",  df.burned_area_ha.min() > 0)
check("duration >= 1",    df.duration_days.min() >= 1)
check("valid causes",     df.ignition_cause.isin(["lightning","human","arson","unknown"]).all())

check("lat within climate grid", df.lat.between(climate_lat.min(), climate_lat.max()).all(),
      f"range {df.lat.min():.2f}–{df.lat.max():.2f}")
check("lon within climate grid", df.lon.between(climate_lon.min(), climate_lon.max()).all(),
      f"range {df.lon.min():.2f}–{df.lon.max():.2f}")

print("\n=== land_use.tif ===")
with rasterio.open(DATA / "land_use.tif") as src:
    lu = src.read(1)
    bounds = src.bounds
    crs    = src.crs

check("dtype uint8",      lu.dtype == np.uint8)
check("no missing values",not np.any(lu == 0))
check("CRS is WGS84",     crs.to_epsg() == 4326)
check("bounds match climate grid",
      abs(bounds.left - climate_lon.min()) < 0.5 and abs(bounds.right - climate_lon.max()) < 0.5 and
      abs(bounds.bottom - climate_lat.min()) < 0.5 and abs(bounds.top - climate_lat.max()) < 0.5)

print("\n=== socioeconomic.csv ===")
df_s = pd.read_csv(DATA / "socioeconomic.csv")

check("row count > 0",    len(df_s) > 0, f"{len(df_s)} rows")
check("columns present",  {"municipality_id","population","gdp_per_capita","infrastructure_density"}.issubset(df_s.columns))
check("no missing values",df_s.isnull().sum().sum() == 0)
check("population > 0",   df_s.population.min() > 0)
check("gdp > 0",          df_s.gdp_per_capita.min() > 0)
check("infra > 0",        df_s.infrastructure_density.min() > 0)
check("unique IDs",       df_s.municipality_id.nunique() == len(df_s))

print(f"\n{'='*40}")
if errors:
    print(f"FAILED {len(errors)} check(s): {', '.join(errors)}")
else:
    print("All checks passed.")
