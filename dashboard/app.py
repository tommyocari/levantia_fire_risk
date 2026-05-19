import sys
from pathlib import Path
from datetime import date

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.model import compute_risk_map
from utils.data import load_municipalities, load_socio, load_fires, load_muni_grid
from utils.plots import risk_map_figure, fire_history_figure, report_pdf

st.set_page_config(page_title="Fire Risk Dashboard", layout="wide")
st.title("Wildfire Risk Map")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    selected_date = st.date_input(
        "Date",
        value=date(2023, 7, 15),
        min_value=date(2014, 1, 1),
        max_value=date(2023, 12, 31),
    )
    st.subheader("Climate scenario")
    temp_offset     = st.slider("Temperature offset (°C)", 0,   4,   0, step=1)
    humidity_offset = st.slider("Humidity offset (%)",    -30,  0,   0, step=5)

    st.header("Municipality")
    socio = load_socio()
    selected_muni = st.selectbox(
        "Select municipality",
        options=socio["municipality_id"].tolist(),
    )

# ── Risk map ──────────────────────────────────────────────────────────────────
prob_grid     = compute_risk_map(str(selected_date), temp_offset=temp_offset, humidity_offset=humidity_offset)
municipalities = load_municipalities()

st.pyplot(risk_map_figure(prob_grid, municipalities, selected_date))

col1, col2, col3 = st.columns(3)
col1.metric("Min",  f"{prob_grid.min():.3f}")
col2.metric("Mean", f"{prob_grid.mean():.3f}")
col3.metric("Max",  f"{prob_grid.max():.3f}")

# ── Municipality profile ──────────────────────────────────────────────────────
st.subheader(f"{selected_muni} — Profile")

fires     = load_fires()
muni_grid = load_muni_grid()

muni_data  = socio[socio["municipality_id"] == selected_muni].iloc[0]
muni_fires = fires[fires["municipality_id"] == selected_muni]
muni_cells = muni_grid == selected_muni
avg_risk   = prob_grid[muni_cells].mean() if muni_cells.any() else float("nan")

col1, col2, col3 = st.columns(3)
col1.metric("Population",             f"{int(muni_data['population']):,}")
col2.metric("GDP per capita",         f"€{int(muni_data['gdp_per_capita']):,}")
col3.metric("Infrastructure density", f"{muni_data['infrastructure_density']:.2f}")

col4, col5 = st.columns(2)
col4.metric("Historical fire events", len(muni_fires))
col5.metric("Average risk today",     f"{avg_risk:.3f}")

st.subheader("Fire history")
st.pyplot(fire_history_figure(muni_fires, fires))

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Export")
pdf_bytes = report_pdf(
    prob_grid, municipalities, muni_fires, fires, muni_data,
    selected_muni, selected_date, avg_risk, temp_offset, humidity_offset,
)
st.download_button(
    label="Download PDF report",
    data=pdf_bytes,
    file_name=f"fire_risk_{selected_muni}_{selected_date}.pdf",
    mime="application/pdf",
)
