# Dataset — socioeconomic.csv

## What I learned

1. **Structure** — one row per municipality with three variables: population (lognormal, independent), GDP per capita (lognormal, independent from population), and infrastructure density (lognormal, correlated with population with r ≈ 0.82 in log-space). The strong correlation between infrastructure and population is realistic: large cities have dense road networks, small villages do not. The noise term accounts for exceptions.

2. **Linking to the other datasets** — this file has no geographical coordinates (`lat`, `lon`), which makes it a non-spatial table. To join it with the climate grid or the fire events (which are spatial), there are a few options:
   - **Add coordinates**: give each municipality a centroid lat/lon and spatially join it to the grid cells or fire events that fall within its territory.
   - **Crosswalk table**: create a separate mapping table that links `municipality_id` to a set of grid cell indices.