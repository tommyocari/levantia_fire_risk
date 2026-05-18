import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

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
# Load — fire events only
# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA / "features.csv", parse_dates=["date"])
df = df[df["target"] == 1].reset_index(drop=True)

X = df.drop(columns=DROP_COLS)
y = np.log1p(df["burned_area_ha"])

nan_mask = X.isnull().any(axis=1)
X, y     = X[~nan_mask], y[~nan_mask]
dates    = df.loc[~nan_mask, "date"]
print(f"Fire events after NaN drop: {len(X)}")

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
# Step 1 — Hyperparameter tuning via GridSearchCV + PredefinedSplit
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
    ("reg", XGBRegressor(n_estimators=200, random_state=42, verbosity=0)),
])

param_grid = {
    "reg__max_depth":        [3, 5], # how many splits
    "reg__learning_rate":    [0.01, 0.05, 0.1], # how much to learn from previous one
    "reg__subsample":        [0.7, 1.0], # fraction of training data re-randomized at each iteration
    "reg__colsample_bytree": [0.7, 1.0], # same idea but on columns
}

search = GridSearchCV(pipe, param_grid, cv=ps, scoring="neg_root_mean_squared_error", refit=True)
search.fit(X_tv, y_tv)

print("\nHyperparameter search — top 10 combinations (sorted by val RMSE):")
results = (
    pd.DataFrame({
        "rmse":            -search.cv_results_["mean_test_score"],
        "max_depth":       [p["reg__max_depth"]        for p in search.cv_results_["params"]],
        "learning_rate":   [p["reg__learning_rate"]    for p in search.cv_results_["params"]],
        "subsample":       [p["reg__subsample"]        for p in search.cv_results_["params"]],
        "colsample_bytree":[p["reg__colsample_bytree"] for p in search.cv_results_["params"]],
    })
    .sort_values("rmse")
    .reset_index(drop=True)
)
print(results.head(10).to_string(index=False))
print(f"\nBest params : {search.best_params_}")
print(f"Best val RMSE: {-search.best_score_:.4f}")

best_params = {k.replace("reg__", ""): v for k, v in search.best_params_.items()}

# --------------------------------------------------------------------------- #
# Step 2 — Learning curve with early stopping → find optimal n_estimators
# --------------------------------------------------------------------------- #
# Fit preprocessor on train only, transform train and val separately
pre = search.best_estimator_["pre"]
X_train_t = pre.transform(X_train)
X_val_t   = pre.transform(X_val)

xgb_diag = XGBRegressor(
    **best_params,
    n_estimators=500,
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0,
)
xgb_diag.fit(
    X_train_t, y_train,
    eval_set=[(X_train_t, y_train), (X_val_t, y_val)],
    verbose=False,
)

train_rmse = xgb_diag.evals_result()["validation_0"]["rmse"]
val_rmse   = xgb_diag.evals_result()["validation_1"]["rmse"]
best_round = xgb_diag.best_iteration # put the best n_estimators for final testing

fig_lc, ax_lc = plt.subplots(figsize=(9, 4))
ax_lc.plot(train_rmse, label="Train RMSE", color="steelblue")
ax_lc.plot(val_rmse,   label="Val RMSE",   color="crimson")
ax_lc.axvline(best_round, color="grey", linestyle="--", linewidth=0.8,
              label=f"Best round: {best_round}")
ax_lc.set_xlabel("Boosting round")
ax_lc.set_ylabel("RMSE  (log scale)")
ax_lc.set_title("XGBoost learning curve — train vs val RMSE")
ax_lc.legend()
fig_lc.tight_layout()
fig_lc.savefig(DATA.parent / "notebooks" / "xgb_learning_curve.png", dpi=150)
plt.show()
print(f"Early stopping: optimal n_estimators = {best_round}")

# --------------------------------------------------------------------------- #
# Step 3 — Refit final model: best hyperparams + best_round as n_estimators
# --------------------------------------------------------------------------- #
final_model = Pipeline([
    ("pre", preprocessor),
    ("reg", XGBRegressor(**best_params, n_estimators=best_round, random_state=42, verbosity=0)),
])
final_model.fit(X_tv, y_tv)
print(f"\nFinal model: {best_params} | n_estimators={best_round}")

# --------------------------------------------------------------------------- #
# Step 4 — Final evaluation on test set — touched only once
# --------------------------------------------------------------------------- #
pred_test  = final_model.predict(X_test)
rmse_test  = np.sqrt(mean_squared_error(y_test, pred_test))
mae_test   = mean_absolute_error(y_test, pred_test)
r2_test    = r2_score(y_test, pred_test)

print(f"\nFinal evaluation on test set (2022-2023):")
print(f"  RMSE : {rmse_test:.4f}")
print(f"  MAE  : {mae_test:.4f}")
print(f"  R²   : {r2_test:.4f}")
print(f"  (log scale — target is log1p(burned_area_ha))")

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.scatter(y_test, pred_test, alpha=0.6, s=20, color="steelblue")
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", linewidth=0.8)
ax2.set_xlabel("Actual log(burned area)")
ax2.set_ylabel("Predicted log(burned area)")
ax2.set_title(f"XGBoost tuned — test 2022-2023\nRMSE={rmse_test:.3f} | R²={r2_test:.3f}")
fig2.tight_layout()
fig2.savefig(DATA.parent / "notebooks" / "regressor_final.png", dpi=150)
plt.show()
print("\nPlots saved to notebooks/")

# --------------------------------------------------------------------------- #
# Save model
# --------------------------------------------------------------------------- #
models_dir = DATA.parent / "models"
models_dir.mkdir(exist_ok=True)
joblib.dump(final_model, models_dir / "xgb_regressor.pkl")
print(f"Model saved to {models_dir / 'xgb_regressor.pkl'}")
