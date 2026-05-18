import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay,
)

DATA = Path(__file__).parent.parent / "data"

DROP_COLS = ["date", "lat", "lon", "burned_area_ha", "duration_days",
             "ignition_cause", "municipality_id", "target"]

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

train_mask = dates.dt.year <= 2020
val_mask   = dates.dt.year == 2021
test_mask  = dates.dt.year >= 2022

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

X_train = X_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_val   = X_val[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_test  = X_test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

print(f"Train : {len(X_train)} rows (2014-2020)")
print(f"Val   : {len(X_val)}   rows (2021)")
print(f"Test  : {len(X_test)}  rows (2022-2023)")

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
preprocessor = ColumnTransformer([
    ("num", Pipeline([("scale", StandardScaler())]),                                             NUMERIC_FEATURES),
    ("cat", Pipeline([("enc",   OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),  CATEGORICAL_FEATURES),
])

# --------------------------------------------------------------------------- #
# Hyperparameter tuning — PredefinedSplit keeps the temporal order
# Train indices: -1  |  Val indices: 0
# --------------------------------------------------------------------------- #
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])

test_fold = np.concatenate([
    np.full(len(X_train), -1),
    np.zeros(len(X_val)),
])
ps = PredefinedSplit(test_fold)

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {"clf__C": np.logspace(-3, 3, 10).tolist()}

search = GridSearchCV(pipe, param_grid, cv=ps, scoring="roc_auc", refit=True)
search.fit(X_tv, y_tv)

print("\nHyperparameter search results:")
for mean, params in zip(search.cv_results_["mean_test_score"], search.cv_results_["params"]):
    print(f"  C={params['clf__C']:<8.4f} — AUC-ROC: {mean:.4f}")

print(f"\nBest C: {search.best_params_['clf__C']}  (val AUC: {search.best_score_:.4f})")

# --------------------------------------------------------------------------- #
# Plot: val AUC vs C
# --------------------------------------------------------------------------- #
c_values = [p["clf__C"] for p in search.cv_results_["params"]]
auc_values = search.cv_results_["mean_test_score"]

fig_c, ax_c = plt.subplots(figsize=(7, 4))
ax_c.plot(c_values, auc_values, marker="o", color="steelblue")
ax_c.axvline(search.best_params_["clf__C"], color="crimson", linestyle="--",
             label=f"Best C={search.best_params_['clf__C']}")
ax_c.set_xscale("log")
ax_c.set_xlabel("C  (regularisation strength, log scale)")
ax_c.set_ylabel("AUC-ROC on val (2021)")
ax_c.set_title("Hyperparameter tuning — Logistic Regression")
ax_c.legend()
fig_c.tight_layout()
fig_c.savefig(DATA.parent / "notebooks" / "lr_tuning.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------- #
# Final evaluation on test set — touched only once
# --------------------------------------------------------------------------- #
best_model = search.best_estimator_
proba_test = best_model.predict_proba(X_test)[:, 1] # predict
auc_test   = roc_auc_score(y_test, proba_test) # evaluate AOC
brier_test = brier_score_loss(y_test, proba_test) # evaluate Brier score 

print(f"\nFinal evaluation on test set (2022-2023):")
print(f"  AUC-ROC     : {auc_test:.4f}")
print(f"  Brier score : {brier_test:.4f}")

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
fig_pr,  ax_pr  = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_predictions(y_test, proba_test, name=f"LR tuned (AUC={auc_test:.3f})", ax=ax_roc)
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax_roc.set_title("ROC curve — tuned Logistic Regression (test 2022-2023)")
fig_roc.tight_layout()
fig_roc.savefig(DATA.parent / "notebooks" / "roc_final.png", dpi=150)

PrecisionRecallDisplay.from_predictions(y_test, proba_test, name="LR tuned", ax=ax_pr)
baseline = y_test.mean()
ax_pr.axhline(baseline, color="black", linewidth=0.8, linestyle="--", label=f"Random (precision={baseline:.2f})")
ax_pr.legend()
ax_pr.set_title("PR curve — tuned Logistic Regression (test 2022-2023)")
fig_pr.tight_layout()
fig_pr.savefig(DATA.parent / "notebooks" / "pr_final.png", dpi=150)

plt.show()
print("\nPlots saved to notebooks/")
