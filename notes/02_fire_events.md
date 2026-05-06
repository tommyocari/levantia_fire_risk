# Dataset — fire_events.csv

## What I learned

1. **Climate-weighted spatial sampling** — to place fires in realistic locations, the logic is: for each sampled event date, look up the climate grid for that day, identify which cells have high temperature and low humidity, build a probability distribution over the 50×50 grid from that risk score = normalized temperature + (1-normalized humidity), and sample a lat/lon from it. Cells that are hot and dry on that day are much more likely to be picked as the fire location.

2. **Correlated lognormals via a shared underlying normal** — burned area and duration are both lognormal and should be correlated (bigger fires burn longer). The way to encode this is: generate the log of burned area as a normal draw, then build the log of duration as a linear function of that same normal plus independent Gaussian noise. Exponentiating both gives two lognormals that share a common source of variation — the noise term controls how loose the correlation is.
