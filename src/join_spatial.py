import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import KDTree


LAND_USE_LABELS = {1: "urban", 2: "forest", 3: "shrubland", 4: "agricultural", 5: "water"}


def join_land_use(df, tif_path):
    """Add land_use column by looking up each fire event in the land use raster."""
    with rasterio.open(tif_path) as src:
        lu        = src.read(1)
        transform = src.transform

    rows, cols = rasterio.transform.rowcol(transform, df["lon"].values, df["lat"].values)
    rows = np.clip(rows, 0, lu.shape[0] - 1)
    cols = np.clip(cols, 0, lu.shape[1] - 1)
    df["land_use"] = [LAND_USE_LABELS.get(int(c), "unknown") for c in lu[rows, cols]]
    return df


def join_socioeconomic(df, socio):
    """Assign each fire event to the nearest municipality centroid, then merge attributes."""
    _, idx = KDTree(socio[["lat", "lon"]].values).query(df[["lat", "lon"]].values)
    df["municipality_id"] = socio["municipality_id"].iloc[idx].values

    cols = [c for c in socio.columns if c not in ("lat", "lon", "municipality_id")]
    df   = df.merge(socio[["municipality_id"] + cols], on="municipality_id", how="left")
    return df
