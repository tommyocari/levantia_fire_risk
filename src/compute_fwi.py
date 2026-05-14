import numpy as np


def compute_fwi(df):
    """Add FWI columns from day-of and 7-day rolling climate values.

    Simplified proxy:
        FWI = (T/30) * ((100-H)/100) * (1 + W/20) * exp(-P/10)
    """
    def _fwi(T, H, W, P):
        return (T / 30) * ((100 - H) / 100) * (1 + W / 20) * np.exp(-P / 10)

    df["fwi"]    = _fwi(df["temperature"],    df["humidity"],    df["wind_speed"],    df["precipitation"]).clip(lower=0)
    df["fwi_7d"] = _fwi(df["temperature_7d"], df["humidity_7d"], df["wind_speed_7d"], df["precipitation_7d"]).clip(lower=0)
    return df
