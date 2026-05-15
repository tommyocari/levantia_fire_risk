import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path

from spatial_join        import join_climate, join_land_use, join_socioeconomic
from running_average     import join_rolling
from compute_fwi         import compute_fwi
from compute_anomalies   import compute_anomalies
from negative_generation import generate_negatives

DATA         = Path(__file__).parent.parent / "data"
OUT          = DATA / "features.csv"
CLIMATE_VARS = ["temperature", "humidity", "wind_speed", "precipitation", "ndvi"]

print("Loading fire events...")
df_pos = pd.read_csv(DATA / "fire_events.csv", parse_dates=["date"])
df_pos["target"] = 1
print(f"  {len(df_pos)} events, date range: {df_pos['date'].min().date()} -> {df_pos['date'].max().date()}")

print("Generating negative samples...")
df_neg = generate_negatives(df_pos)
print(f"  {len(df_neg)} negative samples generated")

df = pd.concat([df_pos, df_neg], ignore_index=True)
print(f"  Total: {len(df)} rows (target=1: {df_pos['target'].sum()}, target=0: {(df['target']==0).sum()})")

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
