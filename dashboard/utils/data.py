from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
import streamlit as st

DATA = Path(__file__).parent.parent.parent / "data"


@st.cache_data
def load_fires() -> pd.DataFrame:
    fires = pd.read_csv(DATA / "fire_events.csv", parse_dates=["date"])
    municipalities = gpd.read_file(DATA / "municipalities.geojson")
    fires_gdf = gpd.GeoDataFrame(
        fires,
        geometry=gpd.points_from_xy(fires.lon, fires.lat),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        fires_gdf, municipalities[["municipality_id", "geometry"]],
        how="left", predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    fires["municipality_id"] = joined["municipality_id"].values
    return fires


@st.cache_data
def load_socio() -> pd.DataFrame:
    return pd.read_csv(DATA / "socioeconomic.csv")


@st.cache_data
def load_municipalities() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA / "municipalities.geojson")


@st.cache_resource
def load_climate_grid() -> xr.Dataset:
    return xr.open_dataset(DATA / "climate_grid.nc")


@st.cache_resource
def load_land_use():
    return rasterio.open(DATA / "land_use.tif")


@st.cache_data
def load_muni_grid() -> np.ndarray:
    """Return a (NY, NX) array of municipality_id strings aligned with the climate grid."""
    ds = xr.open_dataset(DATA / "climate_grid.nc")
    lats, lons = ds.lat.values, ds.lon.values
    NY, NX = len(lats), len(lons)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    df = pd.DataFrame({"lat": lat_grid.ravel(), "lon": lon_grid.ravel()})

    municipalities = gpd.read_file(DATA / "municipalities.geojson")
    grid_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        grid_gdf, municipalities[["municipality_id", "geometry"]],
        how="left", predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["municipality_id"].values.reshape(NY, NX)
