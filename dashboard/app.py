import sys
from pathlib import Path
from datetime import date

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.model import compute_risk_map
from utils.data import load_municipalities
from utils.plots import risk_map_figure

st.set_page_config(page_title="Fire Risk Dashboard", layout="wide")
st.title("Wildfire Risk Map")

with st.sidebar:
    st.header("Settings")
    selected_date = st.date_input(
        "Date",
        value=date(2023, 7, 15),
        min_value=date(2014, 1, 1),
        max_value=date(2023, 12, 31),
    )

prob_grid = compute_risk_map(str(selected_date))
municipalities = load_municipalities()

st.pyplot(risk_map_figure(prob_grid, municipalities, selected_date))

col1, col2, col3 = st.columns(3)
col1.metric("Min", f"{prob_grid.min():.3f}")
col2.metric("Mean", f"{prob_grid.mean():.3f}")
col3.metric("Max", f"{prob_grid.max():.3f}")
