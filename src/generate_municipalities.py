import numpy as np
import geopandas as gpd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from pathlib import Path

SEED    = 42
N       = 300
LAT_MIN, LAT_MAX = 36.0, 46.0
LON_MIN, LON_MAX =  6.0, 18.0

OUT_PATH = Path(__file__).parent.parent / "data" / "municipalities.geojson"

rng = np.random.default_rng(SEED)

# Random seed points (lon, lat)
points = np.column_stack([
    rng.uniform(LON_MIN, LON_MAX, N),
    rng.uniform(LAT_MIN, LAT_MAX, N),
])

# Mirror points across each boundary so all Voronoi regions are finite
mirrors = np.vstack([
    points,
    np.column_stack([2 * LON_MIN - points[:, 0], points[:, 1]]),  # left
    np.column_stack([2 * LON_MAX - points[:, 0], points[:, 1]]),  # right
    np.column_stack([points[:, 0], 2 * LAT_MIN - points[:, 1]]),  # bottom
    np.column_stack([points[:, 0], 2 * LAT_MAX - points[:, 1]]),  # top
])

vor  = Voronoi(mirrors)
bbox = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

polygons = []
for i in range(N):
    region   = vor.regions[vor.point_region[i]]
    vertices = vor.vertices[region]
    poly     = Polygon(vertices).intersection(bbox)
    polygons.append(poly)

gdf = gpd.GeoDataFrame(
    {"municipality_id": [f"MUN_{i:03d}" for i in range(N)]},
    geometry=polygons,
    crs="EPSG:4326",
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
gdf.to_file(OUT_PATH, driver="GeoJSON")
print(f"Written {len(gdf)} municipality polygons -> {OUT_PATH}")
