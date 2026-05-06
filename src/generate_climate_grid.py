import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.ndimage import gaussian_filter

SEED = 42
NX, NY = 50, 50
SPATIAL_SIGMA = 2.0  # smoothing radius in grid cells
START, END = "2014-01-01", "2023-12-31"
OUT_PATH = Path(__file__).parent.parent / "data" / "climate_grid.nc"

rng = np.random.default_rng(SEED)
dates = pd.date_range(START, END, freq="D")
NT = len(dates)

# Lat/lon grid centred on a generic Mediterranean region
lat = np.linspace(36.0, 46.0, NY, dtype="f4")
lon = np.linspace(6.0, 18.0, NX, dtype="f4")

# Spatial base fields — smooth background using sine waves
x_idx = np.arange(NX)
y_idx = np.arange(NY)
xx, yy = np.meshgrid(x_idx, y_idx)
spatial_base = np.sin(np.pi * xx / NX) * np.cos(np.pi * yy / NY)  # (NY, NX)

# Day-of-year seasonal signal
doy = np.array([d.timetuple().tm_yday for d in dates], dtype="f4")
seasonal = np.sin(2 * np.pi * (doy - 80) / 365)  # peaks in summer

def make_field(seasonal_amp, base_mean, spatial_scale, noise_std):
    """Build (NT, NY, NX) array with seasonal + spatial + spatially correlated noise."""
    s = seasonal_amp * seasonal[:, None, None]
    b = base_mean + spatial_scale * spatial_base[None, :, :]
    n = rng.normal(0, noise_std, (NT, NY, NX)).astype("f4")
    n = gaussian_filter(n, sigma=(0, SPATIAL_SIGMA, SPATIAL_SIGMA))
    return (s + b + n).astype("f4")

temperature   = make_field(seasonal_amp=12.0, base_mean=18.0, spatial_scale=4.0, noise_std=1.5)  # °C
humidity      = make_field(seasonal_amp=-15.0, base_mean=60.0, spatial_scale=8.0, noise_std=5.0)  # %
wind_speed    = make_field(seasonal_amp=1.5, base_mean=5.0, spatial_scale=2.0, noise_std=1.0)     # m/s
precipitation = make_field(seasonal_amp=-1.5, base_mean=2.0, spatial_scale=1.0, noise_std=2.5)  # mm/day

humidity = np.clip(humidity, 0, 100).astype("f4")
wind_speed = np.clip(wind_speed, 0, None).astype("f4")
precipitation = np.clip(precipitation, 0, None).astype("f4")

# NDVI: peaks in spring (doy ~120), lower in summer drought and winter
ndvi_seasonal = np.sin(2 * np.pi * (doy - 60) / 365)  # peaks ~late April
ndvi_noise = gaussian_filter(rng.normal(0, 0.03, (NT, NY, NX)), sigma=(0, SPATIAL_SIGMA, SPATIAL_SIGMA))
ndvi = (
    0.25 * ndvi_seasonal[:, None, None]
    + 0.45 + 0.15 * spatial_base[None, :, :]
    + ndvi_noise
).astype("f4")
ndvi = np.clip(ndvi, -1.0, 1.0).astype("f4")

## writing the dataset on the output

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def make_var(data, units, long_name):
    return xr.Variable(("time", "lat", "lon"), data, attrs={"units": units, "long_name": long_name})

ds = xr.Dataset(
    {
        "temperature":   make_var(temperature,   "°C",      "Near-surface air temperature"),
        "humidity":      make_var(humidity,      "%",       "Relative humidity"),
        "wind_speed":    make_var(wind_speed,    "m s-1",   "Near-surface wind speed"),
        "precipitation": make_var(precipitation, "mm day-1","Daily precipitation"),
        "ndvi":          make_var(ndvi,          "1",       "Normalized Difference Vegetation Index"),
    },
    coords={"time": dates, "lat": lat, "lon": lon},
    attrs={
        "title":       "Synthetic climate grid 2014-2023",
        "institution": "Levantia Fire Risk",
        "source":      "generate_data.py",
        "history":     f"Created {pd.Timestamp.now().isoformat()}",
        "Conventions": "CF-1.8",
    },
)

# compress the dataset (not essential)
encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
ds.to_netcdf(OUT_PATH, encoding=encoding)

print(f"Written {NT} time steps × {NY}×{NX} grid → {OUT_PATH}")
print(f"  temperature : {temperature.min():.1f} – {temperature.max():.1f} °C")
print(f"  humidity    : {humidity.min():.1f} – {humidity.max():.1f} %")
print(f"  wind_speed  : {wind_speed.min():.1f} – {wind_speed.max():.1f} m/s")
print(f"  precipitation: {precipitation.min():.1f} – {precipitation.max():.1f} mm/day")
print(f"  ndvi         : {ndvi.min():.2f} – {ndvi.max():.2f}")