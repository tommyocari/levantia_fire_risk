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

# Reduced set — correlated features removed (see notes/14_feature_selection_classifier.md)
NUMERIC_FEATURES = [
    "temperature", "humidity",
    "temperature_anom", "wind_speed_anom", "precipitation_anom",
    "population", "gdp_per_capita", "infrastructure_density",
]
CATEGORICAL_FEATURES = ["land_use"]

# --------------------------------------------------------------------------- #
# Load & split
# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA / "features.csv", parse_dates=["date"])

X = df.drop(columns=DROP_COLS)
y = df["target"]

nan_mask = X.isnull().any(axis=1)
X, y     = X[~nan_mask], y[~nan_mask]
dates    = df.loc[~nan_mask, "date"]
print(f"Dropped {nan_mask.sum()} rows with missing feature values")
print(f"Remaining — target=0: {(y == 0).sum()} | target=1: {(y == 1).sum()} | total: {len(y)}")

# Temporal split — no shuffle to avoid data leakage
# Test set (2022-2023) is kept for final evaluation only, not touched here
train_mask = dates.dt.year <= 2020
val_mask   = dates.dt.year == 2021

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]

X_train = X_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_val   = X_val[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

print(f"Train : {len(X_train)} rows (2014-2020)")
print(f"Val   : {len(X_val)}   rows (2021)")

# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #
preprocessor = ColumnTransformer([
    ("num", Pipeline([("scale", StandardScaler())]),                                             NUMERIC_FEATURES),
    ("cat", Pipeline([("enc",   OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),  CATEGORICAL_FEATURES),
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

val_aucs = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)

    proba_val    = pipe.predict_proba(X_val)[:, 1]
    auc          = roc_auc_score(y_val, proba_val)
    brier        = brier_score_loss(y_val, proba_val)
    val_aucs[name] = auc

    print(f"\n{name}")
    print(f"  Val (2021) — AUC-ROC: {auc:.4f} | Brier: {brier:.4f}")

    RocCurveDisplay.from_predictions(y_val, proba_val, name=f"{name} (AUC={auc:.3f})", ax=ax_roc)
    PrecisionRecallDisplay.from_predictions(y_val, proba_val, name=name, ax=ax_pr)

ax_roc.set_title("ROC curves — fire occurrence classifier (val 2021)")
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
fig_roc.tight_layout()
fig_roc.savefig(DATA.parent / "notebooks" / "roc_curves.png", dpi=150)

baseline = y_val.mean()
ax_pr.axhline(baseline, color="black", linewidth=0.8, linestyle="--", label=f"Random (precision={baseline:.2f})")
ax_pr.legend()
ax_pr.set_title("Precision-Recall curves — fire occurrence classifier (val 2021)")
fig_pr.tight_layout()
fig_pr.savefig(DATA.parent / "notebooks" / "pr_curves.png", dpi=150)

plt.show()
print("\nPlots saved to notebooks/")

# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
best_name = max(val_aucs, key=val_aucs.get)
print(f"\nBest model on val (2021): {best_name} — AUC-ROC: {val_aucs[best_name]:.4f}")
