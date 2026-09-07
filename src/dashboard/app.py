import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8010"

st.set_page_config(page_title="FleetMind", layout="wide")
st.title("FleetMind - Fault Diagnosis Pipeline")

try:
    health = requests.get(f"{API_BASE}/health", timeout=3).json()
    st.success(f"Backend connected - database status: {health['database']}")
except requests.exceptions.RequestException:
    st.error(
        "Cannot reach the FastAPI backend at "
        f"{API_BASE}. Start it first: `uvicorn src.api.main:app --reload`"
    )
    st.stop()

st.header("Simulated Depots (real CWRU load levels)")
depots = requests.get(f"{API_BASE}/depots").json()
depots_df = pd.DataFrame(depots)
st.dataframe(depots_df, use_container_width=True)

st.header("Fault Class Distribution")
dist = requests.get(f"{API_BASE}/fault-distribution").json()
if dist:
    dist_df = pd.DataFrame(dist).set_index("label")
    st.bar_chart(dist_df["n_windows"])
    st.caption(
        "This is the real class balance across ingested signal windows - the "
        "starting point for deciding which class(es) need TimeGAN augmentation."
    )
else:
    st.info("No signal windows ingested yet - run scripts/ingest_cwru_data.py.")

st.header("Raw Signal Viewer")
fault_type = st.selectbox(
    "Fault type", ["Normal", "InnerRace", "OuterRace", "Ball"]
)
experiments = requests.get(f"{API_BASE}/experiments", params={"fault_type": fault_type}).json()

if not experiments:
    st.info(f"No ingested experiments with fault_type={fault_type} yet.")
else:
    exp_options = {f"experiment {e['id']} - {e['source_file']}": e["id"] for e in experiments}
    chosen_label = st.selectbox("Experiment", list(exp_options.keys()))
    experiment_id = exp_options[chosen_label]

    windows = requests.get(f"{API_BASE}/experiments/{experiment_id}/windows", params={"limit": 5}).json()
    if not windows:
        st.info("This experiment has no ingested windows yet.")
    else:
        window_options = {f"window {w['window_index']}": w["id"] for w in windows}
        chosen_window_label = st.selectbox("Window", list(window_options.keys()))
        window_id = window_options[chosen_window_label]
        window = requests.get(f"{API_BASE}/signal-window/{window_id}").json()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(window["raw_values"])
        ax.set_title(f"{window['label']} - window {window['window_index']}")
        ax.set_xlabel("sample index")
        ax.set_ylabel("amplitude")
        st.pyplot(fig)
