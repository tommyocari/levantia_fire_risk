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

# --- Dates: seasonal bias toward summer (July–August peak) ---
all_days = pd.date_range(START, END, freq="D")
doy = np.array([d.timetuple().tm_yday for d in all_days])
# weight peaks around day 210 (late July)
weights = np.exp(-0.5 * ((doy - 210) / 60) ** 2)
weights /= weights.sum()
event_dates = rng.choice(all_days, size=N_EVENTS, replace=True, p=weights)
event_dates = pd.DatetimeIndex(event_dates)

# --- Spatial location: weighted by high temperature + low humidity ---
climate = xr.open_dataset(CLIMATE_PATH)
grid_lat = climate.lat.values
grid_lon = climate.lon.values
NY, NX = len(grid_lat), len(grid_lon)

lats, lons = [], []
for date in event_dates:
    temp = climate["temperature"].sel(time=date, method="nearest").values  # (NY, NX)
    hum  = climate["humidity"].sel(time=date, method="nearest").values

    # normalise each field to [0, 1] then combine
    temp_norm = (temp - temp.min()) / (temp.max() - temp.min() + 1e-9)
    hum_norm  = (hum  - hum.min())  / (hum.max()  - hum.min()  + 1e-9)
    risk = temp_norm + (1 - hum_norm)   # high temp + low humidity = high risk

    prob = risk / risk.sum()
    flat_idx = rng.choice(NY * NX, p=prob.flatten())
    lat_idx, lon_idx = np.unravel_index(flat_idx, (NY, NX)) # extracting from a flat matrix and then recomputing the two indexes
    lats.append(round(grid_lat[lat_idx], 4))
    lons.append(round(grid_lon[lon_idx], 4))

climate.close()
lats = np.array(lats)
lons = np.array(lons)

# --- Burned area: lognormal (most fires small, few very large) ---
# median ~50 ha, heavy right tail up to ~50000 ha
burned_area = np.exp(rng.normal(loc=3.9, scale=1.4, size=N_EVENTS))
burned_area = np.clip(burned_area, 1, 60000).round(1)

# --- Duration: lognormal, correlated with burned area ---
log_duration = 0.4 * np.log(burned_area) + rng.normal(0, 0.6, N_EVENTS)
duration_days = np.clip(np.exp(log_duration - 1.5).round().astype(int), 1, 60)

# --- Ignition cause ---
ignition_cause = rng.choice(CAUSES, size=N_EVENTS, p=CAUSE_PROBS)

df = pd.DataFrame({
    "date":            event_dates.strftime("%Y-%m-%d"),
    "lat":             lats,
    "lon":             lons,
    "burned_area_ha":  burned_area,
    "duration_days":   duration_days,
    "ignition_cause":  ignition_cause,
})

df = df.sort_values("date").reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"Written {len(df)} fire events → {OUT_PATH}")
print(f"  date range      : {df.date.min()} – {df.date.max()}")
print(f"  burned_area_ha  : {df.burned_area_ha.min():.1f} – {df.burned_area_ha.max():.1f} (median {df.burned_area_ha.median():.1f})")
print(f"  duration_days   : {df.duration_days.min()} – {df.duration_days.max()} (median {df.duration_days.median():.0f})")
print(f"  ignition_cause  :\n{df.ignition_cause.value_counts().to_string()}")
