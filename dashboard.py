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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR = os.getenv("GOLD_DIR", os.path.join(BASE_DIR, "data", "gold", "chassis_comfort_delta"))
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
            
        # Ensure columns exist for backward compatibility across schema evolutions
        for col_name, default_val in [
            ('weighted_acc_z', 0.0),
            ('vdv_acc_z', 0.0),
            ('crest_factor', 1.0),
            ('dominant_freq', 0.0),
            ('shock_event', 'NOMINAL'),
            ('iso_comfort_tier', 'Comfortable'),
        ]:
            if col_name not in df.columns:
                df[col_name] = default_val
            
        # Parse time and sort chronologically for line charts
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Numeric cleanup for NaN/Inf
        for num_col in ['weighted_acc_z', 'vdv_acc_z', 'crest_factor', 'comfort_score', 'rms_acc_z', 'dominant_freq']:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0.0)
        
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

    c1, c2, c3, c4, c5 = st.columns(5)
    
    # ISO 2631-1 Thresholds & Color Coding
    w_acc = latest['weighted_acc_z']
    vdv_val = latest.get('vdv_acc_z', 0.0)
    cf_val = latest.get('crest_factor', 1.0)
    shock_flag = latest.get('shock_event', 'NOMINAL')
    tier_label = latest.get('iso_comfort_tier', 'Comfortable')
    dom_freq = latest.get('dominant_freq', 0.0)

    if pd.isna(w_acc) or not np.isfinite(w_acc):
        status, color, emoji = "Initializing...", "#888888", "⚪"
    elif w_acc < 0.315:
        status, color, emoji = "Comfortable", "#00FF00", "🟢"
    elif w_acc < 0.63:
        status, color, emoji = "A little uncomfortable", "#FFFF00", "🟡"
    elif w_acc < 1.0:
        status, color, emoji = "Fairly uncomfortable", "#FF9900", "🟠"
    elif w_acc < 1.6:
        status, color, emoji = "Uncomfortable", "#FF6600", "🟠"
    elif w_acc < 2.5:
        status, color, emoji = "Very Uncomfortable", "#FF3300", "🔴"
    else:
        status, color, emoji = "Extremely Uncomfortable", "#CC0000", "🛑"

    with c1:
        st.metric("Comfort Score", f"{latest['comfort_score']:.1f}%", f"{latest['comfort_score'] - prev['comfort_score']:.1f}%")
        st.markdown(f"<p style='color:{color}; font-weight:bold;'>{emoji} {status}</p>", unsafe_allow_html=True)
    with c2:
        st.metric("ISO 2631 Weighted (a_w)", f"{w_acc:.3f} m/s²", f"{w_acc - prev['weighted_acc_z']:.4f}", delta_color="inverse")
        st.caption(f"Tier: **{tier_label}**")
    with c3:
        prev_vdv = prev.get('vdv_acc_z', vdv_val)
        st.metric("Vibration Dose Value (VDV)", f"{vdv_val:.3f} m/s¹·⁷⁵", f"{vdv_val - prev_vdv:.4f}", delta_color="inverse")
        st.caption("Transient shock dose (4th power)")
    with c4:
        cf_color = "#FF4444" if cf_val > 9.0 else "#00CC66"
        st.metric("Crest Factor (Peak/RMS)", f"{cf_val:.2f}")
        st.markdown(f"<p style='color:{cf_color}; font-weight:bold; font-size:12px;'>{'⚠️ ' + shock_flag if cf_val > 9.0 else '✅ ' + shock_flag}</p>", unsafe_allow_html=True)
    with c5:
        st.metric("Speed / Dominant Freq", f"{latest['avg_speed']:.1f} km/h")
        st.caption(f"Peak Frequency: **{dom_freq:.1f} Hz**")

    st.divider()

    # Main Grid: Trends over Time
    row1_left, row1_right = st.columns(2)

    with row1_left:
        st.subheader("📊 Real-Time Comfort Index")
        # Line chart for Comfort Score
        fig_score = px.line(df_viz, x="start_time", y="comfort_score", 
                            template="plotly_dark", color_discrete_sequence=["#00D4FF"])
        fig_score.update_layout(yaxis_range=[0, 105], margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_score, use_container_width=True)

    with row1_right:
        st.subheader("⚡ Continuous Vibration (a_w) vs. Shock Metric (VDV)")
        # Multi-trace chart for Weighted vs VDV vs RMS
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["rms_acc_z"], name="Raw RMS", fill='tozeroy', line=dict(color="#636EFA", width=1.5)))
        fig_acc.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["weighted_acc_z"], name="ISO Weighted (a_w)", line=dict(color="#FF7F0E", width=2.5)))
        if 'vdv_acc_z' in df_viz.columns:
            fig_acc.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["vdv_acc_z"], name="VDV Shock (m/s^1.75)", line=dict(color="#00FFAA", width=2, dash="dot")))
        
        # Add ISO Threshold Lines
        fig_acc.add_hline(y=0.315, line_dash="dash", line_color="green", annotation_text="Comfortable (<0.315)")
        fig_acc.add_hline(y=0.63, line_dash="dash", line_color="yellow", annotation_text="Mild Uncomfort. (0.63)")
        fig_acc.add_hline(y=1.25, line_dash="dash", line_color="red", annotation_text="Uncomfortable (1.25)")
        
        fig_acc.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_acc, use_container_width=True)

    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.subheader("📈 ISO 2631 Weighted Acc. & VDV Trends")
        fig_vib = go.Figure()
        fig_vib.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["weighted_acc_z"], name="Weighted Acc (a_w)", fill='tozeroy', line=dict(color="#FF7F0E")))
        if 'vdv_acc_z' in df_viz.columns:
            fig_vib.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["vdv_acc_z"], name="VDV", line=dict(color="#00FFAA", width=2)))
        fig_vib.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_vib, use_container_width=True)

    with row2_right:
        st.subheader("🎯 Crest Factor vs Speed (Shock Profiling)")
        # Scatter for Crest Factor vs Speed
        y_scatter = "crest_factor" if "crest_factor" in df_viz.columns else "weighted_acc_z"
        fig_corr = px.scatter(df_viz, x="avg_speed", y=y_scatter, 
                              size="rms_acc_z", color="comfort_score", 
                              hover_data=["test_id", "shock_event"] if "shock_event" in df_viz.columns else ["test_id"],
                              template="plotly_dark", color_continuous_scale="Viridis")
        fig_corr.add_hline(y=9.0, line_dash="dash", line_color="red", annotation_text="ISO Shock Threshold (CF > 9)")
        fig_corr.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Historical Data Table
    with st.expander("Extended History Data (View All Samples)"):
        st.write(f"Displaying all {len(df_viz)} processed windows for **{selected_vehicle}** - Test: **{selected_test}**")
        cols_to_show = [c for c in ['start_time', 'vehicle_id', 'comfort_score', 'iso_comfort_tier', 'weighted_acc_z', 'vdv_acc_z', 'crest_factor', 'shock_event', 'dominant_freq', 'avg_speed', 'rms_acc_z', 'peak_acc_z'] if c in df_viz.columns]
        st.dataframe(df_viz[cols_to_show].sort_values("start_time", ascending=False), use_container_width=True)

else:
    st.warning(" No data available for the selected flight/vehicle criteria.")

# Footer
st.markdown("---")
st.caption("Vehicle Dynamics & Comfort AI Lab | Real-time Streaming Architecture")
