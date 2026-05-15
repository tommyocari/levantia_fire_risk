# WildFireWatch

End-to-end wildfire risk data pipeline for **Levantia**, a fictional Mediterranean territory. The project generates a realistic synthetic dataset, validates it, and lays the ground for a risk modelling product that non-technical stakeholders can use directly.

This is a portfolio project exploring geospatial data engineering with Python. Developed with the help of [Claude Code](https://claude.ai/code).

---

## Project overview

| Dataset | Format | Description |
|---|---|---|
| `climate_grid.nc` | NetCDF | 50×50 spatial grid, daily 2014–2023. Variables: temperature, humidity, wind speed, precipitation, NDVI |
| `fire_events.csv` | CSV | ~800 wildfire events with date, location, burned area, duration, ignition cause |
| `land_use.tif` | GeoTIFF | Static 50×50 land use raster (urban / forest / shrubland / agricultural / water) |
| `socioeconomic.csv` | CSV | 300 municipalities with population, GDP per capita, infrastructure density |

All spatial datasets share the same geographic extent (lat 36–46°N, lon 6–18°E).

---

## Repository structure

```
levantia_fire_risk/
├── src/
│   ├── generate_climate_grid.py   # generates climate_grid.nc
│   ├── generate_fire_events.py    # generates fire_events.csv
│   ├── generate_land_use.py       # generates land_use.tif
│   ├── generate_socioeconomic.py  # generates socioeconomic.csv
│   ├── validate_data.py           # checks integrity of all datasets
│   └── report_data.py             # prints and saves a data quality report
├── notebooks/
│   └── 01_explore_datasets.ipynb  # interactive exploration of all datasets
├── notes/                         # learning notes per dataset
├── data/                          # generated files (not tracked by git)
├── generate_data.sh               # runs all generators in order
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/tommyocari/levantia_fire_risk.git
cd levantia_fire_risk
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Generate all datasets

```bash
./generate_data.sh
```

This runs all five generators in order and writes the files to `data/`.

### 2. Run the enrichment pipeline

```bash
cd src
python pipeline.py
```

The pipeline joins each fire event with its climate values (day-of and 7-day rolling mean), computes the Fire Weather Index and climatological anomalies, assigns land use and municipality, and writes `data/fire_events_enriched.csv`.

### 3. Visualise the data

Open the notebooks in Jupyter:

```bash
jupyter notebook
```

| Notebook | Content |
|---|---|
| `notebooks/01_visualise_datasets.ipynb` | Raw datasets — climate grid, fire events, land use, municipalities |
| `notebooks/02_socioeconomic_map.ipynb` | Socioeconomic attributes visualised as choropleth maps |

### 4. Validate and report

```bash
python src/validate_data.py
python src/report_data.py
```

---

## Milestones

| Milestone | Status | Description |
|---|---|---|
| 1 — Dataset generation | Done | Synthetic climate grid, fire events, land use, municipalities, socioeconomic |
| 2 — Enrichment pipeline | Done | Climate join, rolling means, FWI, anomalies, land use and municipality join |

---

## Tech stack

- **xarray** — NetCDF reading and writing
- **rasterio** — GeoTIFF reading and writing
- **pandas / numpy** — tabular data and numerical generation
- **scipy** — Gaussian spatial autocorrelation
- **matplotlib** — visualisation
