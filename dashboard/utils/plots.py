import io

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(all_years)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_ylabel("Fire events")
    fig.tight_layout()
    return fig


def report_pdf(
    prob_grid: np.ndarray,
    municipalities: gpd.GeoDataFrame,
    muni_fires: pd.DataFrame,
    all_fires: pd.DataFrame,
    muni_data: pd.Series,
    selected_muni: str,
    selected_date,
    avg_risk: float,
    temp_offset: float,
    humidity_offset: float,
) -> bytes:
    year_min  = int(all_fires["date"].dt.year.min())
    year_max  = int(all_fires["date"].dt.year.max())
    all_years = list(range(year_min, year_max + 1))
    global_max = (
        all_fires
        .groupby([all_fires["date"].dt.year, all_fires["municipality_id"]])
        .size()
        .max()
    )
    counts = muni_fires.groupby(muni_fires["date"].dt.year).size().reindex(all_years, fill_value=0)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        gs  = GridSpec(3, 2, figure=fig, height_ratios=[0.6, 5, 2.8], hspace=0.45, wspace=0.35)

        # ── Header ────────────────────────────────────────────────────────────
        ax_h = fig.add_subplot(gs[0, :])
        ax_h.axis("off")
        if temp_offset or humidity_offset:
            scenario = f"+{temp_offset} °C  /  {humidity_offset:+d} % humidity"
        else:
            scenario = "baseline (no climate offset)"
        ax_h.text(0, 0.95, "Wildfire Risk Report",
                  fontsize=18, fontweight="bold", va="top", transform=ax_h.transAxes)
        ax_h.text(0, 0.55, f"Municipality: {selected_muni}   |   Date: {selected_date}",
                  fontsize=11, va="top", transform=ax_h.transAxes)
        ax_h.text(0, 0.15, f"Climate scenario: {scenario}",
                  fontsize=10, color="gray", va="top", transform=ax_h.transAxes)

        # ── Risk map ──────────────────────────────────────────────────────────
        ax_map = fig.add_subplot(gs[1, 0])
        im = ax_map.imshow(
            prob_grid, origin="lower", extent=[6.0, 18.0, 36.0, 46.0],
            cmap="YlOrRd", vmin=0, vmax=1,
        )
        municipalities.boundary.plot(ax=ax_map, color="black", linewidth=0.4)
        plt.colorbar(im, ax=ax_map, label="Fire probability", shrink=0.8)
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.set_title("Risk map", fontweight="bold", loc="left")

        # ── Municipality profile ──────────────────────────────────────────────
        ax_m = fig.add_subplot(gs[1, 1])
        ax_m.axis("off")
        ax_m.set_title("Municipality Profile", fontweight="bold", loc="left")
        rows = [
            ("Population",             f"{int(muni_data['population']):,}"),
            ("GDP per capita",         f"€{int(muni_data['gdp_per_capita']):,}"),
            ("Infrastructure density", f"{muni_data['infrastructure_density']:.2f}"),
            ("Historical fire events", str(len(muni_fires))),
            ("Average risk today",     f"{avg_risk:.3f}"),
        ]
        for i, (label, value) in enumerate(rows):
            y = 0.88 - i * 0.18
            ax_m.text(0, y,        label, fontsize=9,  color="gray",      transform=ax_m.transAxes)
            ax_m.text(0, y - 0.07, value, fontsize=13, fontweight="bold", transform=ax_m.transAxes)

        # ── Fire history ──────────────────────────────────────────────────────
        ax_f = fig.add_subplot(gs[2, :])
        ax_f.bar(all_years, counts.values, color="#d62728")
        ax_f.set_xlim(year_min - 0.5, year_max + 0.5)
        ax_f.set_ylim(0, global_max)
        ax_f.set_xticks(all_years)
        ax_f.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_f.spines["top"].set_visible(False)
        ax_f.spines["right"].set_visible(False)
        ax_f.set_xlabel("Year")
        ax_f.set_ylabel("Fire events")
        ax_f.set_title("Fire History", fontweight="bold", loc="left")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buf.seek(0)
    return buf.read()
