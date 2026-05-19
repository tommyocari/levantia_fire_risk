import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
import streamlit as st

DATA = Path(__file__).parent.parent.parent / "data"


@st.cache_data
def load_fire_events() -> pd.DataFrame:
    return pd.read_csv(DATA / "fire_events.csv", parse_dates=["date"])


@st.cache_data
def load_socioeconomic() -> pd.DataFrame:
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
