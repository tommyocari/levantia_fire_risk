import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

DATA = Path(__file__).parent.parent / "data"

DROP_COLS = ["date", "lat", "lon", "burned_area_ha", "duration_days",
             "ignition_cause", "municipality_id", "target"]

# Reduced set — correlated features removed
NUMERIC_FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "temperature_anom",
    "humidity_anom",
    "wind_speed_anom",
    "precipitation_anom",
    "population",
    "gdp_per_capita",
    "infrastructure_density",
]
CATEGORICAL_FEATURES = ["land_use"]

# --------------------------------------------------------------------------- #
# Load full dataset — no split, fit is only for coefficient inspection
# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA / "features.csv", parse_dates=["date"])
X  = df.drop(columns=DROP_COLS)
y  = df["target"]

nan_mask = X.isnull().any(axis=1)
X, y     = X[~nan_mask], y[~nan_mask]

X = X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

# --------------------------------------------------------------------------- #
# Fit on all data
# --------------------------------------------------------------------------- #
preprocessor = ColumnTransformer([
    ("num", Pipeline([("scale", StandardScaler())]),                                             NUMERIC_FEATURES),
    ("cat", Pipeline([("enc",   OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),  CATEGORICAL_FEATURES),
])

lr = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
lr.fit(X, y)

# --------------------------------------------------------------------------- #
# Extract and clean coefficients
# --------------------------------------------------------------------------- #
feature_names = lr["pre"].get_feature_names_out()
coefs         = lr["clf"].coef_[0]

coef_df = (
    pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=True)
)
coef_df["feature"] = coef_df["feature"].str.replace(r"^(num__|cat__)", "", regex=True)

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 6))

colors = ["crimson" if c > 0 else "steelblue" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient  (red = increases fire probability, blue = decreases)")
ax.set_title("Logistic Regression — reduced feature set coefficients", fontsize=13)
plt.tight_layout()

out = DATA.parent / "notebooks" / "logistic_weights.png"
fig.savefig(out, dpi=150)
plt.show()
print(f"Plot saved to {out}")
