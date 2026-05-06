# NetCDF datasets and xarray

## What is a NetCDF dataset

A NetCDF dataset is a dataset that has an underlying grid (space and time) with multiple variables defined on top of that grid. The grid is made of named coordinate axes (e.g. `time`, `lat`, `lon`) and the variables (e.g. `temperature`, `humidity`) are arrays indexed by those axes.

## Opening and exploring with xarray

```python
import xarray as xr

ds = xr.open_dataset('climate_grid.nc')
ds   # prints a full summary: dimensions, coordinates, variables, attributes
```

`open_dataset` both opens the file and gives a complete overview of its structure in one call.

## Accessing a variable

Always access variables like a dictionary — never rely on attribute access:

```python
ds['temperature']   # returns a DataArray of shape (time, lat, lon)
```

## Selecting along coordinates

The key operation is `.sel()` — it lets you restrict the data along any coordinate by value rather than by index:

```python
# single point in time → spatial map (lat, lon)
ds['temperature'].sel(time='2018-07-15')

# single grid cell → time series
ds['temperature'].sel(lat=41.0, lon=12.0, method='nearest')

# single cell, single year
ds['temperature'].sel(lat=41.0, lon=12.0, method='nearest').sel(time='2018')
```

`method='nearest'` is important when the exact coordinate value is not in the grid.

## Plotting

xarray has a built-in `.plot()` that automatically labels axes and adds colorbars:

```python
ds['temperature'].sel(time='2018-07-15').plot(cmap='RdYlBu_r')   # 2D heatmap
ds['temperature'].sel(lat=41.0, lon=12.0, method='nearest').plot()  # time series
```

## Aggregations

Average over space first, then group — always reduce the large dimensions before grouping:

```python
spatial_mean = ds['temperature'].mean(('lat', 'lon'))    # (3652,)
monthly_mean = spatial_mean.groupby('time.month').mean() # (12,)
```

---

# GeoTIFF datasets and rasterio

## What is a GeoTIFF dataset

A GeoTIFF is a raster dataset with a 2D spatial grid only — no time dimension. It is the industry standard for storing static geographical features (land use, elevation, soil type, etc.). Each file can contain multiple bands (features), each stored as a separate 2D array.

## Opening and reading with rasterio

```python
import rasterio

with rasterio.open('land_use.tif') as src:
    band = src.read(1)        # read the 1st band as a numpy array (NY, NX)
    transform = src.transform # maps pixel indices to geographic coordinates
    crs = src.crs             # coordinate reference system (e.g. WGS84)
    bounds = src.bounds       # geographic extent (left, bottom, right, top)
    tags = src.tags()         # metadata tags
```

`src.read(n)` reads the n-th band (1-indexed). If the file has multiple bands call `src.read()` with no argument to get all of them as `(n_bands, NY, NX)`.

## Selecting a spatial subset (windowed reading)

Rasterio supports reading only a portion of the grid using a `Window`, which avoids loading the full file into memory:

```python
from rasterio.windows import from_bounds

with rasterio.open('land_use.tif') as src:
    # define a geographic bounding box to read
    window = from_bounds(
        left=8.0, bottom=40.0, right=14.0, top=44.0,
        transform=src.transform
    )
    subset = src.read(1, window=window)   # reads only that spatial region
    subset_transform = src.window_transform(window)  # updated transform for the subset
```

## Converting pixel index to geographic coordinates

```python
import rasterio.transform

row, col = 10, 20
lon, lat = rasterio.transform.xy(transform, row, col)
```

## Geographic referencing

In GeoTIFF the transform is embedded directly in the file header — no convention or metadata attributes needed. Any tool opening the file automatically knows the geographic coordinates of each pixel. This is unlike NetCDF, where geographic meaning is declared through the CF convention and stored in explicit coordinate variables.
