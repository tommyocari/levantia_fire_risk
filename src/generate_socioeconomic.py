import numpy as np
import pandas as pd
from pathlib import Path

SEED             = 42
N_MUNICIPALITIES = 300
OUT_PATH         = Path(__file__).parent.parent / "data" / "socioeconomic.csv"

rng = np.random.default_rng(SEED)

# --- Population: lognormal ---
log_pop    = rng.normal(loc=8.5, scale=1.8, size=N_MUNICIPALITIES)
population = np.clip(np.exp(log_pop), 200, 2_000_000).round().astype(int)

# --- GDP per capita: independent lognormal ---
log_gdp        = rng.normal(loc=9.8, scale=0.4, size=N_MUNICIPALITIES)
gdp_per_capita = np.clip(np.exp(log_gdp), 5_000, 80_000).round().astype(int)

# --- Infrastructure density: correlated with population (more people -> more roads) ---
log_pop_c              = log_pop - log_pop.mean()
log_infra              = 0.4 * log_pop_c + rng.normal(loc=1.5, scale=0.5, size=N_MUNICIPALITIES)
infrastructure_density = np.clip(np.exp(log_infra), 0.5, 30.0).round(2)

df = pd.DataFrame({
    "municipality_id":        [f"MUN_{i:03d}" for i in range(N_MUNICIPALITIES)],
    "population":             population,
    "gdp_per_capita":         gdp_per_capita,
    "infrastructure_density": infrastructure_density,
})

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"Written {len(df)} municipalities -> {OUT_PATH}")
print(f"  population            : {df.population.min()} - {df.population.max()} (median {df.population.median():.0f})")
print(f"  gdp_per_capita        : {df.gdp_per_capita.min()} - {df.gdp_per_capita.max()} (median {df.gdp_per_capita.median():.0f})")
print(f"  infrastructure_density: {df.infrastructure_density.min()} - {df.infrastructure_density.max()} (median {df.infrastructure_density.median():.2f})")
