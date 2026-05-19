import subprocess
import sys


STEPS = [
    ("climate_grid.nc",      "src/generate_climate_grid.py"),
    ("fire_events.csv",      "src/generate_fire_events.py"),
    ("land_use.tif",         "src/generate_land_use.py"),
    ("socioeconomic.csv",    "src/generate_socioeconomic.py"),
    ("municipalities.geojson", "src/generate_municipalities.py"),
]


print("\n=== Generating all datasets ===")
for i, (label, script) in enumerate(STEPS, 1):
    print(f"[{i}/{len(STEPS)}] {label}")
    subprocess.run([sys.executable, script], check=True)
    print(i,label,script)
print("=== Done ===")