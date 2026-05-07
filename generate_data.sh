#!/bin/bash
set -e

echo ""
echo "=== Generating all datasets ==="

echo "[1/4] climate_grid.nc"
python src/generate_climate_grid.py

echo "[2/4] fire_events.csv"
python src/generate_fire_events.py

echo "[3/4] land_use.tif"
python src/generate_land_use.py

echo "[4/4] socioeconomic.csv"
python src/generate_socioeconomic.py

echo "=== Done ==="
