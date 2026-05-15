import numpy as np
import pandas as pd


def generate_negatives(df_pos: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate negative (no-fire) samples matching the size and bounds of df_pos.
    - 70% of dates are drawn from summer (June-August), 30% from the rest of the year.
    - Exact (date, lat, lon) matches with real fire events are removed.
    """
    rng = np.random.default_rng(seed=seed)
    n = len(df_pos)

    lat_min, lat_max = df_pos["lat"].min(), df_pos["lat"].max()
    lon_min, lon_max = df_pos["lon"].min(), df_pos["lon"].max()
    date_min, date_max = df_pos["date"].min(), df_pos["date"].max()

    # sample more from summer months to reflect seasonal fire patterns, but still include some from other months
    all_dates = pd.date_range(date_min, date_max, freq="D")
    summer_dates = all_dates[all_dates.month.isin([6, 7, 8])]
    other_dates  = all_dates[~all_dates.month.isin([6, 7, 8])]

    n_summer = int(n * 0.7)
    n_other  = n - n_summer

    random_dates = np.concatenate([
        rng.choice(summer_dates, size=n_summer, replace=True),
        rng.choice(other_dates,  size=n_other,  replace=True),
    ])
    rng.shuffle(random_dates)

    df_neg = pd.DataFrame({
        "date":           pd.to_datetime(random_dates),
        "lat":            rng.uniform(lat_min, lat_max, n).round(4),
        "lon":            rng.uniform(lon_min, lon_max, n).round(4),
        "burned_area_ha": 0.0,
        "duration_days":  0,
        "ignition_cause": pd.NA,
        "target":         0,
    })

    # Remove any generated negatives that exactly match a positive event (same date, lat, lon)
    fire_set = set(zip(df_pos["date"].astype(str), df_pos["lat"], df_pos["lon"]))
    mask = df_neg.apply(lambda r: (str(r["date"]), r["lat"], r["lon"]) in fire_set, axis=1)
    n_collisions = mask.sum()
    print(f"  Negative generation: {n_collisions} candidate(s) matched a real fire event and were removed")
    df_neg = df_neg[~mask].reset_index(drop=True)

    return df_neg
