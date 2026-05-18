import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA = Path(__file__).parent.parent / "data"

DROP_COLS = ["date", "lat", "lon", "burned_area_ha", "duration_days",
             "ignition_cause", "municipality_id", "target"]

# Same reduced independent feature set as classifier
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
print(f"Fire events: {len(df)} rows")

X = df.drop(columns=DROP_COLS)
y = np.log1p(df["burned_area_ha"])  # log-transform: burned area is lognormal

nan_mask = X.isnull().any(axis=1)
X, y     = X[~nan_mask], y[~nan_mask]
dates    = df.loc[~nan_mask, "date"]
print(f"Dropped {nan_mask.sum()} rows with missing feature values")

# Temporal split
train_mask = dates.dt.year <= 2020
val_mask   = dates.dt.year == 2021

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]

X_train = X_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_val   = X_val[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

print(f"Train : {len(X_train)} rows (2014-2020)")
print(f"Val   : {len(X_val)}   rows (2021)")

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
preprocessor = ColumnTransformer([
    ("num", Pipeline([("scale", StandardScaler())]),                                             NUMERIC_FEATURES),
    ("cat", Pipeline([("enc",   OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),  CATEGORICAL_FEATURES),
])

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
models = {
    "Linear Regression": Pipeline([
        ("pre", preprocessor),
        ("reg", LinearRegression()),
    ]),
    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("reg", XGBRegressor(n_estimators=200, learning_rate=0.05,
                             random_state=42, verbosity=0)),
    ]),
}

# --------------------------------------------------------------------------- #
# Train, evaluate, plot
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

val_rmses = {}

for ax, (name, pipe) in zip(axes, models.items()):
    pipe.fit(X_train, y_train)

    pred_val = pipe.predict(X_val)
    rmse     = np.sqrt(mean_squared_error(y_val, pred_val))
    mae      = mean_absolute_error(y_val, pred_val)
    r2       = r2_score(y_val, pred_val)
    val_rmses[name] = rmse

    print(f"\n{name}")
    print(f"  Val (2021) — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    print(f"  (log scale — target is log1p(burned_area_ha))")

    ax.scatter(y_val, pred_val, alpha=0.6, s=20, color="steelblue")
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "k--", linewidth=0.8)
    ax.set_xlabel("Actual log(burned area)")
    ax.set_ylabel("Predicted log(burned area)")
    ax.set_title(f"{name}\nRMSE={rmse:.3f} | R²={r2:.3f}")

plt.suptitle("Burned area regression — val 2021 (log scale)", fontsize=13)
plt.tight_layout()
plt.savefig(DATA.parent / "notebooks" / "regressor_val.png", dpi=150)
plt.show()
print("\nPlot saved to notebooks/")

# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
best_name = min(val_rmses, key=val_rmses.get)
print(f"\nBest model on val (2021): {best_name} — RMSE: {val_rmses[best_name]:.4f}")
