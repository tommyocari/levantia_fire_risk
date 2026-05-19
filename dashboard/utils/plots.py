import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import geopandas as gpd


def risk_map_figure(prob_grid: np.ndarray, municipalities: gpd.GeoDataFrame, date) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        prob_grid,
        origin="lower",
        extent=[6.0, 18.0, 36.0, 46.0],
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    municipalities.boundary.plot(ax=ax, color="black", linewidth=0.5)
    plt.colorbar(im, ax=ax, label="Fire probability")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Fire risk – {date}")
    return fig


def fire_history_figure(muni_fires: pd.DataFrame, all_fires: pd.DataFrame) -> plt.Figure:
    year_min = int(all_fires["date"].dt.year.min())
    year_max = int(all_fires["date"].dt.year.max())
    all_years = list(range(year_min, year_max + 1))

    global_max = (
        all_fires
        .groupby([all_fires["date"].dt.year, all_fires["municipality_id"]])
        .size()
        .max()
    )

    counts = muni_fires.groupby(muni_fires["date"].dt.year).size()
    counts = counts.reindex(all_years, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(all_years, counts.values, color="#d62728")
    ax.set_xlim(year_min - 0.5, year_max + 0.5)
    ax.set_ylim(0, global_max)
    ax.spines["top","right"].set_visible(False)
    ax.set_xticks(all_years)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_ylabel("Fire events")
    fig.tight_layout()
    return fig
