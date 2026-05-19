import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from pathlib import Path

FPS  = 30  # fast playback — ~10 years of daily frames
DATA = Path(__file__).parent.parent / "data"

# --------------------------------------------------------------------------- #
# Load fire events
# --------------------------------------------------------------------------- #
df = pd.read_csv(DATA / "fire_events.csv", parse_dates=["date"])
df["year"] = df["date"].dt.year

# One frame per calendar day — including days with no fire
days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

# Colour each year with a distinct colour from a colormap
years      = sorted(df["year"].unique())
cmap       = cm.get_cmap("tab10", len(years))
year_color = {yr: cmap(i) for i, yr in enumerate(years)}

LAT_MIN, LAT_MAX = 36.0, 46.0
LON_MIN, LON_MAX =  6.0, 18.0

# --------------------------------------------------------------------------- #
# Set up figure
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_facecolor("#f0f0f0")

title   = ax.set_title("")
scatter = ax.scatter([], [], s=20, alpha=0.7, zorder=3)

# Legend: one patch per year
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=year_color[yr], markersize=7, label=str(yr))
    for yr in years
]
ax.legend(handles=legend_handles, title="Year", loc="lower right", fontsize=8)

# --------------------------------------------------------------------------- #
# Precompute — sort once, build cumulative counts and color array upfront
# --------------------------------------------------------------------------- #
df = df.sort_values("date").reset_index(drop=True)
all_coords = np.column_stack([df["lon"].values, df["lat"].values])
all_colors = np.array([year_color[y] for y in df["year"]])

# for each calendar day: how many fire events have occurred up to that day?
cum_counts = np.searchsorted(df["date"].values, days.values, side="right")

# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #
def update(frame):
    n = cum_counts[frame]
    if n > 0:
        scatter.set_offsets(all_coords[:n])
        scatter.set_color(all_colors[:n])
    title.set_text(f"Fire events up to {days[frame].strftime('%Y-%m-%d')}  ({n} total)")
    return scatter, title

ani = animation.FuncAnimation(
    fig, update,
    frames=len(days),
    interval=1000 // FPS,
    blit=True,
)

out_path = DATA.parent / "notebooks" / "fire_events_animated.mp4"
ani.save(out_path, writer=animation.FFMpegWriter(fps=FPS))
plt.close()
print(f"Saved to {out_path}")
