import joblib
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

DATE = "2020-07-15"  # change to any date in 2014-2023

DATA   = Path(__file__).parent.parent / "data"
MODELS = Path(__file__).parent.parent / "models"

NUMERIC_FEATURES     = ["temperature", "humidity",
                         "temperature_anom", "wind_speed_anom", "precipitation_anom",
                         "population", "gdp_per_capita", "infrastructure_density"]
CATEGORICAL_FEATURES = ["land_use"]
LAND_USE_LABELS      = {1: "urban", 2: "forest", 3: "shrubland", 4: "agricultural", 5: "water"}


# --------------------------------------------------------------------------- #
# Module API
# --------------------------------------------------------------------------- #
def load_static_data(data=DATA, models=MODELS):
    """Load everything that does not change across dates.

    Returns a dict with: clf, climate, climatology, df_static, lats, lons, NY, NX.
    Caller is responsible for closing static["climate"] when done.
    """
    clf          = joblib.load(models / "lr_classifier.pkl")
    climate_grid = xr.open_dataset(data / "climate_grid.nc")[
        ["temperature", "humidity", "wind_speed", "precipitation"]
    ].load()

    lats = climate_grid.lat.values
    lons = climate_grid.lon.values
    NY, NX = len(lats), len(lons)

    # montly average to compute anomalies
    monthly_avg = climate_grid.groupby("time.month").mean("time")  # (12, 50, 50)

    # --- static spatial features ---
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    df_static = pd.DataFrame({"lat": lat_grid.ravel(), "lon": lon_grid.ravel()}) # df

    # land use added to df_static
    with rasterio.open(data / "land_use.tif") as src:
        lu        = src.read(1)
        transform = src.transform

    rows, cols = rasterio.transform.rowcol(transform, df_static["lon"].values, df_static["lat"].values)
    rows = np.clip(rows, 0, lu.shape[0] - 1)
    cols = np.clip(cols, 0, lu.shape[1] - 1)
    df_static["land_use"] = [LAND_USE_LABELS.get(int(lu[r, c]), "unknown") for r, c in zip(rows, cols)]

    # socioeconomic features added to df_static via spatial join with municipalities
    municipalities = gpd.read_file(data / "municipalities.geojson")
    socio          = pd.read_csv(data / "socioeconomic.csv")

    grid_gdf = gpd.GeoDataFrame(
        df_static.copy(),
        geometry=gpd.points_from_xy(df_static["lon"], df_static["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(grid_gdf, municipalities[["municipality_id", "geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    df_static["municipality_id"] = joined["municipality_id"].values
    df_static = df_static.merge(
        socio[["municipality_id", "population", "gdp_per_capita", "infrastructure_density"]],
        on="municipality_id", how="left",
    )
    for col in ["population", "gdp_per_capita", "infrastructure_density"]:
        df_static[col] = df_static[col].fillna(socio[col].median())

    # return
    return {
        "clf":          clf,
        "climate_grid": climate_grid,
        "monthly_avg":  monthly_avg,
        "df_static":    df_static,
        "lats":         lats,
        "lons":         lons,
        "NY":           NY,
        "NX":           NX,
    }


def compute_prob_grid(date, static):
    """Return a (NY, NX) array of fire probabilities for the given date."""
    date = pd.Timestamp(date)
    day  = static["climate_grid"].sel(time=date, method="nearest")
    clim_m = static["monthly_avg"].sel(month=date.month)

    temp = day["temperature"].values.ravel()
    hum  = day["humidity"].values.ravel()
    wind = day["wind_speed"].values.ravel()
    prec = day["precipitation"].values.ravel()

    df = static["df_static"].copy()
    df["temperature"]        = temp
    df["humidity"]           = hum
    df["temperature_anom"]   = temp - clim_m["temperature"].values.ravel()
    df["wind_speed_anom"]    = wind - clim_m["wind_speed"].values.ravel()
    df["precipitation_anom"] = prec - clim_m["precipitation"].values.ravel()

    # compute fire probability with the classifier
    prob = static["clf"].predict_proba(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    return prob.reshape(static["NY"], static["NX"])


# --------------------------------------------------------------------------- #
# Standalone use
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Loading static data...")
    static = load_static_data()

    print(f"Computing risk map for {DATE}...")
    prob_grid = compute_prob_grid(DATE, static)
    static["climate_grid"].close()

    print(f"Prob range: {prob_grid.min():.3f} – {prob_grid.max():.3f}  |  mean: {prob_grid.mean():.3f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    mesh = ax.pcolormesh(static["lons"], static["lats"], prob_grid, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(mesh, ax=ax, label="Fire probability")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Predicted fire probability — {DATE}")
    fig.tight_layout()

    out_path = DATA.parent / "notebooks" / f"risk_map_{DATE}.png"
    fig.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved to {out_path}")
