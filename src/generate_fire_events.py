import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

SEED = 42
N_EVENTS = 800
START, END = "2014-01-01", "2023-12-31"
OUT_PATH = Path(__file__).parent.parent / "data" / "fire_events.csv"

LAT_MIN, LAT_MAX = 36.0, 46.0
LON_MIN, LON_MAX = 6.0, 18.0
CLIMATE_PATH = Path(__file__).parent.parent / "data" / "climate_grid.nc"

CAUSES = ["lightning", "human", "arson", "unknown"]
CAUSE_PROBS = [0.25, 0.45, 0.20, 0.10]

rng = np.random.default_rng(SEED)

# --- Dates: seasonal bias toward summer (July-August peak) ---
all_days = pd.date_range(START, END, freq="D")
doy      = np.array([d.timetuple().tm_yday for d in all_days])
weights  = np.exp(-0.5 * ((doy - 210) / 60) ** 2)
weights /= weights.sum()
event_dates = rng.choice(all_days, size=N_EVENTS, replace=True, p=weights)
event_dates = pd.DatetimeIndex(event_dates)

# --- Spatial location: weighted by high temperature + low humidity ---
# Load both fields once into RAM, then select all 800 dates in one vectorised call
climate  = xr.open_dataset(CLIMATE_PATH)[["temperature", "humidity"]].load()
grid_lat = climate.lat.values
grid_lon = climate.lon.values
NY, NX   = len(grid_lat), len(grid_lon)

all_temp = climate["temperature"].sel(
    time=xr.DataArray(event_dates.values, dims="points"), method="nearest"
).values  # (800, NY, NX)
all_hum  = climate["humidity"].sel(
    time=xr.DataArray(event_dates.values, dims="points"), method="nearest"
).values

climate.close()

lats, lons = [], []
for i in range(N_EVENTS):
    temp_norm = (all_temp[i] - all_temp[i].min()) / (all_temp[i].max() - all_temp[i].min() + 1e-9)
    hum_norm  = (all_hum[i]  - all_hum[i].min())  / (all_hum[i].max()  - all_hum[i].min()  + 1e-9)
    risk      = temp_norm + (1 - hum_norm)

    prob     = risk / risk.sum()
    flat_idx = rng.choice(NY * NX, p=prob.flatten())
    lat_idx, lon_idx = np.unravel_index(flat_idx, (NY, NX))
    lats.append(round(grid_lat[lat_idx], 4))
    lons.append(round(grid_lon[lon_idx], 4))

lats = np.array(lats)
lons = np.array(lons)

# --- Burned area: lognormal (most fires small, few very large) ---
burned_area   = np.exp(rng.normal(loc=3.9, scale=1.4, size=N_EVENTS))
burned_area   = np.clip(burned_area, 1, 60000).round(1)

# --- Duration: lognormal, correlated with burned area ---
log_duration  = 0.4 * np.log(burned_area) + rng.normal(0, 0.6, N_EVENTS)
duration_days = np.clip(np.exp(log_duration - 1.5).round().astype(int), 1, 60)

# --- Ignition cause ---
ignition_cause = rng.choice(CAUSES, size=N_EVENTS, p=CAUSE_PROBS)

df = pd.DataFrame({
    "date":           event_dates.strftime("%Y-%m-%d"),
    "lat":            lats,
    "lon":            lons,
    "burned_area_ha": burned_area,
    "duration_days":  duration_days,
    "ignition_cause": ignition_cause,
})

df = df.sort_values("date").reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"Written {len(df)} fire events -> {OUT_PATH}")
print(f"  date range      : {df.date.min()} - {df.date.max()}")
print(f"  burned_area_ha  : {df.burned_area_ha.min():.1f} - {df.burned_area_ha.max():.1f} (median {df.burned_area_ha.median():.1f})")
print(f"  duration_days   : {df.duration_days.min()} - {df.duration_days.max()} (median {df.duration_days.median():.0f})")
print(f"  ignition_cause  :\n{df.ignition_cause.value_counts().to_string()}")
