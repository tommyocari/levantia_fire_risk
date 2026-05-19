import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from risk_map import load_static_data, compute_prob_grid, DATA

START_DATE = "2020-07-01"   # first day of the 30-day window
N_DAYS     = 30
FPS        = 4 # frames per second in the output animation

dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

print("Loading static data...")
static = load_static_data()

print(f"Rendering {N_DAYS} frames ({START_DATE} + {N_DAYS} days)...")

fig, ax = plt.subplots(figsize=(10, 7))
prob0   = compute_prob_grid(dates[0], static)
mesh    = ax.pcolormesh(static["lons"], static["lats"], prob0, cmap="YlOrRd", vmin=0, vmax=1)
plt.colorbar(mesh, ax=ax, label="Fire probability")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
title = ax.set_title("")

def update(frame):
    prob_grid = compute_prob_grid(dates[frame], static)
    mesh.set_array(prob_grid.ravel())
    title.set_text(f"Predicted fire probability — {dates[frame].strftime('%Y-%m-%d')}")
    return mesh, title

ani = animation.FuncAnimation(fig, update, frames=N_DAYS, interval=1000 // FPS, blit=True, repeat_delay=2000)

out_path = DATA.parent / "notebooks" / f"risk_map_animated_{START_DATE}.gif"
ani.save(out_path, writer=animation.PillowWriter(fps=FPS))
plt.close()
print(f"Saved to {out_path}")

static["climate_grid"].close()
