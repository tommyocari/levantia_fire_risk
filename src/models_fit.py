"""Run the full modelling pipeline in order."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent

scripts = [
    ("Task A — Architecture selection (classifier)", "model_classifier_validation.py"),
    ("Task A — Hyperparameter tuning + test (classifier)", "model_classifier_final.py"),
    ("Task B — Architecture selection (regressor)", "model_regressor_validation.py"),
    ("Task B — Hyperparameter tuning + test (regressor)", "model_regressor_final.py"),
]

for label, script in scripts:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, SRC / script], check=False)
    if result.returncode != 0:
        print(f"\nFailed: {script} (exit code {result.returncode})")
        sys.exit(result.returncode)

print("\nAll scripts completed.")
