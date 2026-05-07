# Data validation — validate_data.py

## What I learned

1. **Per-dataset checks** — for each dataset, the validation follows the same structure:
   - Open the file in the correct format (xarray for NetCDF, rasterio for GeoTIFF, pandas for CSV)
   - Check that there is at least one row / cell (`row count > 0`)
   - Check that there are no missing values
   - Check that all expected columns or variables are present
   - Check that numerical values are in a physically realistic range (e.g. humidity in [0, 100], precipitation ≥ 0, duration ≥ 1 day)
   - Check dtypes (e.g. float32 for climate variables, uint8 for land use, datetime for event dates)

2. **Cross-dataset checks** — some checks verify that the datasets are spatially consistent with each other:
   - **GeoTIFF bounds vs climate grid**: the geographic extent of `land_use.tif` must overlap with the lat/lon range of `climate_grid.nc` — checked by comparing the rasterio `bounds` against the min/max of the climate coordinates
   - **Fire events vs climate grid**: the lat/lon of every fire event must fall within the coordinate range of the climate grid — important because fire events were sampled from the climate grid and any mismatch would break spatial joins downstream
