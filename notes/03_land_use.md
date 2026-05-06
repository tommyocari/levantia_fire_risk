# Dataset — land_use.tif

## What I learned

1. **Voronoi tessellation for land use generation** — to create spatially coherent land use patches respecting target proportions, two approaches exist. The first is to generate one smooth random field per class (Gaussian filter on noise) and assign each cell to the class with the highest score, biased by the target proportion. The simpler and more natural approach is Voronoi tessellation: place 60 random seed points on the grid, assign each seed a land use class sampled according to the target proportions (urban 5%, forest 30%, shrubland 30%, agricultural 25%, water 10%), then assign every other cell the class of its nearest seed. This produces contiguous, irregular patches that look like real land use maps without any smoothing needed.
