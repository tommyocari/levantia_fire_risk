# Dataset — climate_grid.nc

## What I learned

1. **Spatial base grid** — the `spatial_base` field (built from sine/cosine waves) represents permanent geographical features of the map: mountains, coastlines, latitude gradients. It is fixed in time and gives each grid cell a distinct climatological mean, so some regions are systematically warmer, drier, etc. than others.

2. **Seasonal signals** — each variable is driven by a sinusoidal cycle over the year. Temperature, humidity, wind, and precipitation peak in summer. NDVI peaks in late April, reflecting Mediterranean vegetation that greens up in spring before the summer drought. The concept is the same for all variables: a sine wave over day-of-year controls how the variable evolves through the seasons.

3. **NetCDF format** — NetCDF (`.nc`) is designed to store multi-dimensional scientific datasets on spatial grids. It keeps data and metadata together in a self-describing file: each variable carries its units, valid range, and long name. Coordinates (time, lat, lon) are stored as named axes so any tool can interpret the structure without external documentation.
