import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from deltalake import DeltaTable

# --- CONFIGURATION ---
GOLD_DIR = "/Users/inesbenyghzer/.gemini/antigravity/scratch/vehicle_comfort_pipeline/data/gold/chassis_comfort_delta"
APP_TITLE = "Real-Time Vehicle Comfort Monitor v3.0"
REFRESH_INTERVAL = 3000  # 3 seconds

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="")

# --- UI STYLING ---
st.markdown("""
<style>
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 8px; border: 1px solid #3e4451; }
    .main { background-color: #0d1117; }
</style>
""", unsafe_allow_html=True)

# --- AUTO-REFRESH ENGINE ---
st_autorefresh(interval=REFRESH_INTERVAL, key="live_refresh_pulse")

# --- DATA ACCESS LAYER ---
@st.cache_data(ttl=3)
def load_comprehensive_data(path):
    """Reads the full history for the selected vehicle session from Delta Lake."""
    if not os.path.exists(path):
        return pd.DataFrame()
    
    try:
        # ACID-compliant read of the transaction log
        dt = DeltaTable(path)
        df = dt.to_pandas()
        
        if df.empty:
            return df
            
        # Ensure new column exists for backward compatibility
        if 'weighted_acc_z' not in df.columns:
            df['weighted_acc_z'] = 0.0
            if 'comfort_vibration' in df.columns:
                df['weighted_acc_z'] = df['comfort_vibration'] # approximation for old data
            
        # Parse time and sort chronologically for line charts
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Final cleanup for NaN/Inf that might đã existed
        df['weighted_acc_z'] = pd.to_numeric(df['weighted_acc_z'], errors='coerce').fillna(0.0)
        df['comfort_score'] = pd.to_numeric(df['comfort_score'], errors='coerce').fillna(100.0)
        
        return df.sort_values("start_time")
    except Exception:
        return pd.DataFrame()

# --- HEADER & SIDEBAR ---
st.title(APP_TITLE)
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Data Storage: Delta Lake (ACID)")

st.sidebar.markdown("### 🛠️ Fleet & Visualization Controls")
df_master = load_comprehensive_data(GOLD_DIR)

if df_master.empty:
    st.info("⏳ Initializing... Waiting for the first Spark commit to Delta Lake.")
    st.stop()

# 1. Dynamic Vehicle Selector
available_vehicles = sorted(df_master["vehicle_id"].unique().tolist())
selected_vehicle = st.sidebar.selectbox(" Select Vehicle ID", available_vehicles)

# 2. Dynamic Test ID Selector (filtered by vehicle)
df_veh = df_master[df_master["vehicle_id"] == selected_vehicle]
available_tests = sorted(df_veh["test_id"].unique().tolist(), reverse=True)
selected_test = st.sidebar.selectbox("📋 Select Test Case", available_tests)

# Filter final dataset for visualization
df_viz = df_veh[df_veh["test_id"] == selected_test]

st.sidebar.divider()
st.sidebar.markdown(f"**Current Version**: `{DeltaTable(GOLD_DIR).version()}`")
st.sidebar.info("The charts below show the FULL history of the selected test run to capture dynamic bumps and spikes.")

# --- DASHBOARD LAYOUT ---

if not df_viz.empty:
    # Top Row: Real-time Counters
    latest = df_viz.iloc[-1]
    prev = df_viz.iloc[-2] if len(df_viz) > 1 else latest

    c1, c2, c3, c4 = st.columns(4)
    
    # ISO 2631-1 Thresholds & Color Coding
    w_acc = latest['weighted_acc_z']
    if pd.isna(w_acc) or not np.isfinite(w_acc):
        status, color, emoji = "Initializing...", "#888888", "⚪"
    elif w_acc < 0.315:
        status, color, emoji = "Comfortable", "#00FF00", "🟢"
    elif w_acc < 0.63:
        status, color, emoji = "Slightly Uncomfortable", "#FFFF00", "🟡"
    elif w_acc < 1.25:
        status, color, emoji = "Uncomfortable", "#FFA500", "🟠"
    else:
        status, color, emoji = "Very Uncomfortable", "#FF0000", "🔴"

    with c1:
        st.metric("Comfort Score", f"{latest['comfort_score']:.1f}%", f"{latest['comfort_score'] - prev['comfort_score']:.1f}%")
        st.markdown(f"<p style='color:{color}; font-weight:bold;'>{emoji} {status}</p>", unsafe_allow_html=True)
    with c2:
        st.metric("ISO 2631 Weighted Acc.", f"{w_acc:.3f} m/s²", f"{w_acc - prev['weighted_acc_z']:.4f}", delta_color="inverse")
    with c3:
        st.metric("RMS Acceleration", f"{latest['rms_acc_z']:.3f} m/s²")
    with c4:
        st.metric("Avg Speed", f"{latest['avg_speed']:.1f} km/h")

    st.divider()

    # Main Grid: Trends over Time
    row1_left, row1_right = st.columns(2)

    with row1_left:
        st.subheader(" Real-Time Comfort Index")
        # Line chart for Comfort Score
        fig_score = px.line(df_viz, x="start_time", y="comfort_score", 
                            template="plotly_dark", color_discrete_sequence=["#00D4FF"])
        fig_score.update_layout(yaxis_range=[0, 105], margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_score, use_container_width=True)

    with row1_right:
        st.subheader("ISO Weighted Acceleration vs RMS")
        # Multi-trace chart for Weighted vs RMS
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["rms_acc_z"], name="Chassis RMS", fill='tozeroy', line=dict(color="#636EFA")))
        fig_acc.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["weighted_acc_z"], name="ISO Weighted (a_w)", line=dict(color="#FF7F0E", width=3)))
        
        # Add ISO Threshold Lines
        fig_acc.add_hline(y=0.315, line_dash="dash", line_color="green", annotation_text="Comfortable")
        fig_acc.add_hline(y=0.63, line_dash="dash", line_color="yellow", annotation_text="Slightly Uncomfort.")
        fig_acc.add_hline(y=1.25, line_dash="dash", line_color="red", annotation_text="Uncomfortable")
        
        fig_acc.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_acc, use_container_width=True)

    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.subheader("ISO 2631 Weighted Acc. Distribution")
        # Area chart for Weighted Acc
        fig_vib = px.area(df_viz, x="start_time", y="weighted_acc_z", 
                          template="plotly_dark", color_discrete_sequence=["#FF7F0E"])
        fig_vib.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_vib, use_container_width=True)

    with row2_right:
        st.subheader(" Weighted Acc. vs Speed")
        # Scatter for correlation
        fig_corr = px.scatter(df_viz, x="avg_speed", y="weighted_acc_z", 
                              size="rms_acc_z", color="comfort_score", 
                              template="plotly_dark", color_continuous_scale="Viridis")
        fig_corr.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Historical Data Table
    with st.expander("Extended History Data (View All Samples)"):
        st.write(f"Displaying all {len(df_viz)} processed windows for **{selected_vehicle}** - Test: **{selected_test}**")
        st.dataframe(df_viz.sort_values("start_time", ascending=False), use_container_width=True)

else:
    st.warning(" No data available for the selected flight/vehicle criteria.")

# Footer
st.markdown("---")
st.caption("Vehicle Dynamics & Comfort AI Lab | Real-time Streaming Architecture")
