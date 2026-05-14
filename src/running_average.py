import xarray as xr


CLIMATE_VARS = ["temperature", "humidity", "wind_speed", "precipitation", "ndvi"]


def join_rolling(df, climate, window=7):
    """Add rolling-mean climate values over `window` days ending on ignition day."""
    climate_nd = climate.rolling(time=window, min_periods=1).mean()
    points_nd  = climate_nd.sel(
        time=xr.DataArray(df["date"].values, dims="points"),
        lat=xr.DataArray(df["lat"].values,   dims="points"),
        lon=xr.DataArray(df["lon"].values,   dims="points"),
        method="nearest"
    )
    for v in CLIMATE_VARS:
        df[f"{v}_{window}d"] = points_nd[v].values
    return df
