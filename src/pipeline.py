import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path

from spatial_join      import join_climate, join_land_use, join_socioeconomic
from running_average   import join_rolling
from compute_fwi       import compute_fwi
from compute_anomalies import compute_anomalies

DATA         = Path(__file__).parent.parent / "data"
OUT          = DATA / "fire_events_enriched.csv"
CLIMATE_VARS = ["temperature", "humidity", "wind_speed", "precipitation", "ndvi"]

print("Loading fire events...")
df = pd.read_csv(DATA / "fire_events.csv", parse_dates=["date"])
print(f"  {len(df)} events, date range: {df['date'].min().date()} -> {df['date'].max().date()}")

print("Loading climate grid...")
nc_path = DATA / "climate_grid.nc"
climate = xr.open_dataset(nc_path)[CLIMATE_VARS].load()
print(f"  Grid: {dict(climate.dims)}")

print("Joining climate values on ignition day...")
df = join_climate(df, climate)
print(f"  Done -- added {CLIMATE_VARS}")

print("Computing 7-day rolling mean...")
df = join_rolling(df, climate, window=7)
print(f"  Done -- added {[f'{v}_7d' for v in CLIMATE_VARS]}")

print("Computing Fire Weather Index...")
df = compute_fwi(df)
print(f"  Done -- fwi range: [{df['fwi'].min():.2f}, {df['fwi'].max():.2f}]")

print("Computing climatological anomalies...")
df = compute_anomalies(df, climate)
print(f"  Done -- added {[f'{v}_anom' for v in CLIMATE_VARS]}")

climate.close()

print("Joining land use...")
df = join_land_use(df, DATA / "land_use.tif")
print(f"  Done -- land use distribution: {df['land_use'].value_counts().to_dict()}")

print("Joining socioeconomic data...")
municipalities = gpd.read_file(DATA / "municipalities.geojson")
socio          = pd.read_csv(DATA / "socioeconomic.csv")
df = join_socioeconomic(df, municipalities, socio)
print(f"  Done -- added population, gdp_per_capita, infrastructure_density")

print("\nWriting output...")
df.to_csv(OUT, index=False)
print(f"  Written {len(df)} rows -> {OUT}")
print(f"  Columns: {list(df.columns)}")
