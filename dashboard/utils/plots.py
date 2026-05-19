import matplotlib.pyplot as plt
import numpy as np
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
