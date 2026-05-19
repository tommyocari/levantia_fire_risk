"""Run the full visualisation pipeline in order."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent

scripts = [
    ("Static risk map (single day)",       "risk_map_merged.py"),
    ("Animated risk map (30-day window)",   "risk_map_animated.py"),
    ("Animated fire events (all years)",    "fire_events_animated.py"),
]

for label, script in scripts:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, SRC / script], check=False)
    if result.returncode != 0:
        print(f"\nFailed: {script} (exit code {result.returncode})")
        sys.exit(result.returncode)

print("\nAll visualisations complete.")
