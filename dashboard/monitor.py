import streamlit as st
import json
import os
import pandas as pd

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
METADATA_PATH = os.path.join(BASE_DIR, "data", "partition_metadata.json")


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None

    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def prepare_dataframe(metadata):
    rows = []

    for cid, data in metadata.items():
        state = data["system_state"]

        rows.append({
            "Client": int(cid),
            "Battery (%)": state["battery"],
            "Latency (ms)": state["latency"],
            "Reliability": state["reliability"]
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Client")


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------

st.set_page_config(page_title="FL Monitoring Dashboard", layout="wide")

st.title("📊 Federated Learning Monitoring Dashboard")
st.markdown("Real-time view of client hardware states and system behavior")

metadata = load_metadata()

if metadata is None:
    st.error("❌ Metadata file not found. Run partition_data.py first.")
    st.stop()

df = prepare_dataframe(metadata)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Clients", len(df))
col2.metric("Avg Battery", f"{df['Battery (%)'].mean():.2f}%")
col3.metric("Avg Latency", f"{df['Latency (ms)'].mean():.2f} ms")

st.divider()

# ---------------------------------------------------
# TABLE VIEW
# ---------------------------------------------------

st.subheader("📋 Client Status Table")
st.dataframe(df, use_container_width=True)

st.divider()

# ---------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------

st.subheader("🔋 Battery Levels")
st.bar_chart(df.set_index("Client")["Battery (%)"])

st.subheader("📡 Latency Distribution")
st.bar_chart(df.set_index("Client")["Latency (ms)"])

st.subheader("🧠 Reliability Scores")
st.bar_chart(df.set_index("Client")["Reliability"])

st.divider()

# ---------------------------------------------------
# AUTO REFRESH BUTTON
# ---------------------------------------------------

if st.button("🔄 Refresh"):
    st.rerun()