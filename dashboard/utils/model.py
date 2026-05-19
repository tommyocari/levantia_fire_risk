import sys
import pickle
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from risk_map import load_static_data, compute_prob_grid  # noqa: E402

MODELS = Path(__file__).parent.parent.parent / "models"


@st.cache_resource
def load_model():
    with open(MODELS / "lr_classifier.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def _get_static():
    return load_static_data()


@st.cache_data
def compute_risk_map(date: str, temp_offset: float = 0.0, humidity_offset: float = 0.0):
    return compute_prob_grid(date, _get_static(), temp_offset=temp_offset, humidity_offset=humidity_offset)
