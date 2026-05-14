import xarray as xr


CLIMATE_VARS = ["temperature", "humidity", "wind_speed", "precipitation", "ndvi"]


def compute_anomalies(df, climate):
    """Add anomaly columns: observed value minus the long-term monthly mean at that location."""
    climatology = climate.groupby("time.month").mean("time")  # (month=12, lat, lon)
    months      = xr.DataArray(df["date"].dt.month.values, dims="points")
    clim_points = climatology.sel(
        month=months,
        lat=xr.DataArray(df["lat"].values, dims="points"),
        lon=xr.DataArray(df["lon"].values, dims="points"),
        method="nearest"
    )
    for v in CLIMATE_VARS:
        df[f"{v}_anom"] = df[v] - clim_points[v].values
    return df
