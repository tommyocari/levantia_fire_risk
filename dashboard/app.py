import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🔥 WildFireWatch")
st.write("Climate risk asessment dashboard for Levantia")

# Slider
temperature_offset = st.slider(
    "Temperature offset (°C)",
    min_value=0.0,
    max_value=4.0,
    value=0.0,
    step=0.5
)

# Plot something simple
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x+temperature_offset))
ax.set_title(f"Effect of +{temperature_offset}°C")
st.pyplot(fig) # display in the app

st.write (f"You have selected: +{temperature_offset}°C")