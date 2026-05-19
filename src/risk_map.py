import joblib
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

DATE = "2020-02-15"  # change to any date in 2014-2023

DATA   = Path(__file__).parent.parent / "data"
MODELS = Path(__file__).parent.parent / "models"

NUMERIC_FEATURES     = ["temperature", "humidity",
                         "temperature_anom", "wind_speed_anom", "precipitation_anom",
                         "population", "gdp_per_capita", "infrastructure_density"]
CATEGORICAL_FEATURES = ["land_use"]
LAND_USE_LABELS      = {1: "urban", 2: "forest", 3: "shrubland", 4: "agricultural", 5: "water"}

# --------------------------------------------------------------------------- #
# Load classifier
# --------------------------------------------------------------------------- #
clf = joblib.load(MODELS / "lr_classifier.pkl")

# --------------------------------------------------------------------------- #
# Step 1 — Climate features for every grid cell on DATE
# --------------------------------------------------------------------------- #
climate = xr.open_dataset(DATA / "climate_grid.nc")[
    ["temperature", "humidity", "wind_speed", "precipitation"]  # ndvi not used by the classifier
].load()

lats = climate.lat.values   # (50,)
lons = climate.lon.values   # (50,)
NY, NX = len(lats), len(lons)

day   = climate.sel(time=DATE, method="nearest")
temp  = day["temperature"].values   # (50, 50)
hum   = day["humidity"].values
wind  = day["wind_speed"].values
prec  = day["precipitation"].values

# --------------------------------------------------------------------------- #
# Step 2 — Anomalies: observed - long-term monthly mean for this month
# --------------------------------------------------------------------------- #
month      = pd.Timestamp(DATE).month
clim_month = climate.groupby("time.month").mean("time").sel(month=month)
temp_anom  = temp - clim_month["temperature"].values
wind_anom  = wind - clim_month["wind_speed"].values
prec_anom  = prec - clim_month["precipitation"].values

climate.close()

# --------------------------------------------------------------------------- #
# Step 3 — Build flat DataFrame: one row per grid cell
# --------------------------------------------------------------------------- #
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")   # (50, 50)

df = pd.DataFrame({
    "lat":                lat_grid.ravel(),
    "lon":                lon_grid.ravel(),
    "temperature":        temp.ravel(),
    "humidity":           hum.ravel(),
    "temperature_anom":   temp_anom.ravel(),
    "wind_speed_anom":    wind_anom.ravel(),
    "precipitation_anom": prec_anom.ravel(),
})

# --------------------------------------------------------------------------- #
# Step 4 — Land use lookup (raster)
# --------------------------------------------------------------------------- #
with rasterio.open(DATA / "land_use.tif") as src:
    lu        = src.read(1)
    transform = src.transform

rows, cols = rasterio.transform.rowcol(transform, df["lon"].values, df["lat"].values)
rows = np.clip(rows, 0, lu.shape[0] - 1)
cols = np.clip(cols, 0, lu.shape[1] - 1)
df["land_use"] = [LAND_USE_LABELS.get(int(lu[r, c]), "unknown") for r, c in zip(rows, cols)]

# --------------------------------------------------------------------------- #
# Step 5 — Socioeconomic join (spatial)
# --------------------------------------------------------------------------- #
municipalities = gpd.read_file(DATA / "municipalities.geojson")
socio          = pd.read_csv(DATA / "socioeconomic.csv")

grid_gdf = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326",
)
joined = gpd.sjoin(grid_gdf, municipalities[["municipality_id", "geometry"]],
                   how="left", predicate="within")

# keep first match per cell in case of duplicates from overlapping polygons
joined = joined[~joined.index.duplicated(keep="first")]
df["municipality_id"] = joined["municipality_id"].values

df = df.merge(
    socio[["municipality_id", "population", "gdp_per_capita", "infrastructure_density"]],
    on="municipality_id", how="left",
)

# Cells outside all municipality polygons get NaN — fill with dataset medians
for col in ["population", "gdp_per_capita", "infrastructure_density"]:
    df[col] = df[col].fillna(socio[col].median())

# --------------------------------------------------------------------------- #
# Step 6 — Predict fire probability
# --------------------------------------------------------------------------- #
X    = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
prob = clf.predict_proba(X)[:, 1]          # (2500,)
prob_grid = prob.reshape(NY, NX)           # (50, 50)

print(f"Date: {DATE}")
print(f"Prob range: {prob.min():.3f} – {prob.max():.3f}  |  mean: {prob.mean():.3f}")

# --------------------------------------------------------------------------- #
# Step 7 — Plot
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 7))
mesh = ax.pcolormesh(lons, lats, prob_grid, cmap="YlOrRd", vmin=0, vmax=1)
plt.colorbar(mesh, ax=ax, label="Fire probability")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Predicted fire probability — {DATE}")
fig.tight_layout()

out_path = DATA.parent / "notebooks" / f"risk_map_{DATE}.png"
fig.savefig(out_path, dpi=150)
plt.show()
print(f"Saved to {out_path}")
