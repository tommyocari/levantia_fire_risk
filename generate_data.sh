#!/bin/bash
set -e

echo ""
echo "=== Generating all datasets ==="

echo "[1/5] climate_grid.nc"
python src/generate_climate_grid.py

echo "[2/5] fire_events.csv"
python src/generate_fire_events.py

echo "[3/5] land_use.tif"
python src/generate_land_use.py

echo "[4/5] socioeconomic.csv"
python src/generate_socioeconomic.py

echo "[5/5] municipalities.geojson"
python src/generate_municipalities.py

echo "=== Done ==="
