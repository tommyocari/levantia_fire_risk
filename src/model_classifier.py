import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from xgboost import XGBClassifier

DATA = Path(__file__).parent.parent / "data"

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
# Excluded: date, lat, lon  -> identifiers, not predictors
# Excluded: burned_area_ha, duration_days, ignition_cause -> fire outcomes
# Excluded: municipality_id -> raw identifier, socioeconomic values used instead
DROP_COLS = ["date", "lat", "lon", "burned_area_ha", "duration_days",
             "ignition_cause", "municipality_id", "target"]

NUMERIC_FEATURES = [
    "temperature", "humidity", "wind_speed", "precipitation", "ndvi",
    "temperature_7d", "humidity_7d", "wind_speed_7d", "precipitation_7d", "ndvi_7d",
    "fwi", "fwi_7d",
    "temperature_anom", "humidity_anom", "wind_speed_anom", "precipitation_anom", "ndvi_anom",
    "population", "gdp_per_capita", "infrastructure_density",
]
CATEGORICAL_FEATURES = ["land_use"]

# --------------------------------------------------------------------------- #
# Load & split
# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA / "features.csv", parse_dates=["date"])

X = df.drop(columns=DROP_COLS)
y = df["target"]

# Drop rows that still have NaN in any feature column (boundary points outside municipality coverage)
nan_mask = X.isnull().any(axis=1)
X, y     = X[~nan_mask], y[~nan_mask]
dates    = df.loc[~nan_mask, "date"]
print(f"Dropped {nan_mask.sum()} rows with missing feature values")
print(f"Remaining — target=0: {(y == 0).sum()} | target=1: {(y == 1).sum()} | total: {len(y)}")

# Temporal split — no shuffle to avoid data leakage
train_mask = dates.dt.year <= 2020
val_mask   = dates.dt.year == 2021
test_mask  = dates.dt.year >= 2022

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Train : {len(X_train)} rows (2014-2020)")
print(f"Val   : {len(X_val)}   rows (2021)")
print(f"Test  : {len(X_test)}  rows (2022-2023)")

# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #
numeric_pipe = Pipeline([
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe,      NUMERIC_FEATURES),
    ("cat", categorical_pipe,  CATEGORICAL_FEATURES),
])

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
models = {
    "Logistic Regression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]),
    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(n_estimators=200, learning_rate=0.05,
                              random_state=42, eval_metric="logloss",
                              verbosity=0)),
    ]),
}

# --------------------------------------------------------------------------- #
# Train, evaluate, plot
# --------------------------------------------------------------------------- #
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
fig_pr,  ax_pr  = plt.subplots(figsize=(7, 6))

for name, pipe in models.items():
    pipe.fit(X_train, y_train)

    print(f"\n{name}")
    for split_name, X_s, y_s in [("Val  (2021)", X_val, y_val), ("Test (2022-23)", X_test, y_test)]:
        proba = pipe.predict_proba(X_s)[:, 1]
        auc   = roc_auc_score(y_s, proba)
        brier = brier_score_loss(y_s, proba)
        print(f"  {split_name} — AUC-ROC: {auc:.4f} | Brier: {brier:.4f}")

    proba_test = pipe.predict_proba(X_test)[:, 1]
    auc_test   = roc_auc_score(y_test, proba_test)
    RocCurveDisplay.from_predictions(y_test, proba_test, name=f"{name} (AUC={auc_test:.3f})", ax=ax_roc)
    PrecisionRecallDisplay.from_predictions(y_test, proba_test, name=name, ax=ax_pr)

ax_roc.set_title("ROC curves — fire occurrence classifier")
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
fig_roc.tight_layout()
fig_roc.savefig(DATA.parent / "notebooks" / "roc_curves.png", dpi=150)

ax_pr.set_title("Precision-Recall curves — fire occurrence classifier")
fig_pr.tight_layout()
fig_pr.savefig(DATA.parent / "notebooks" / "pr_curves.png", dpi=150)

plt.show()
print("\nPlots saved to notebooks/")
