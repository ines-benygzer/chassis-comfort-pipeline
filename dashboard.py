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
            ('weighted_acc_x', 0.0),
            ('weighted_acc_y', 0.0),
            ('total_acc_v', 0.0),
            ('vdv_acc_z', 0.0),
            ('vdv_total', 0.0),
            ('crest_factor', 1.0),
            ('crest_factor_z', 1.0),
            ('dominant_freq', 0.0),
            ('dominant_freq_z', 0.0),
            ('shock_event', 'NOMINAL'),
            ('iso_comfort_tier', 'Comfortable'),
            ('min_tire_load', 1.0),
            ('avg_tire_load', 1.0),
            ('road_holding_tier', 'OPTIMAL (Full Grip)'),
        ]:
            if col_name not in df.columns:
                df[col_name] = default_val
            
        # Parse time and sort chronologically for line charts
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Numeric cleanup for NaN/Inf
        for num_col in ['weighted_acc_z', 'weighted_acc_x', 'weighted_acc_y', 'total_acc_v', 'vdv_acc_z', 'vdv_total', 'crest_factor', 'crest_factor_z', 'comfort_score', 'rms_acc_z', 'dominant_freq', 'dominant_freq_z', 'min_tire_load', 'avg_tire_load']:
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

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    # ISO 2631-1 Multi-Axial Thresholds & Color Coding
    total_av = latest.get('total_acc_v', latest.get('weighted_acc_z', 0.0))
    prev_av = prev.get('total_acc_v', prev.get('weighted_acc_z', total_av))
    w_acc_z = latest.get('weighted_acc_z', 0.0)
    w_acc_x = latest.get('weighted_acc_x', 0.0)
    w_acc_y = latest.get('weighted_acc_y', 0.0)
    vdv_total = latest.get('vdv_total', latest.get('vdv_acc_z', 0.0))
    prev_vdv = prev.get('vdv_total', prev.get('vdv_acc_z', vdv_total))
    cf_val = latest.get('crest_factor_z', latest.get('crest_factor', 1.0))
    shock_flag = latest.get('shock_event', 'NOMINAL')
    tier_label = latest.get('iso_comfort_tier', 'Comfortable')
    dom_freq = latest.get('dominant_freq_z', latest.get('dominant_freq', 0.0))
    tire_load = latest.get('min_tire_load', 1.0)
    holding_tier = latest.get('road_holding_tier', 'OPTIMAL (Full Grip)')

    # Overall Comfort Evaluation based on Point Vibration Total Value (a_v)
    if pd.isna(total_av) or not np.isfinite(total_av):
        status, color, emoji = "Initializing...", "#888888", "⚪"
    elif total_av < 0.315:
        status, color, emoji = "Comfortable", "#00FF00", "🟢"
    elif total_av < 0.63:
        status, color, emoji = "A little uncomfortable", "#FFFF00", "🟡"
    elif total_av < 1.0:
        status, color, emoji = "Fairly uncomfortable", "#FF9900", "🟠"
    elif total_av < 1.6:
        status, color, emoji = "Uncomfortable", "#FF6600", "🟠"
    elif total_av < 2.5:
        status, color, emoji = "Very Uncomfortable", "#FF3300", "🔴"
    else:
        status, color, emoji = "Extremely Uncomfortable", "#CC0000", "🛑"

    with c1:
        st.metric("Comfort Score", f"{latest['comfort_score']:.1f}%", f"{latest['comfort_score'] - prev['comfort_score']:.1f}%")
        st.markdown(f"<p style='color:{color}; font-weight:bold;'>{emoji} {status}</p>", unsafe_allow_html=True)
    with c2:
        st.metric("Total Vibration (a_v)", f"{total_av:.3f} m/s²", f"{total_av - prev_av:.4f}", delta_color="inverse")
        st.caption(f"Z: {w_acc_z:.2f} | X: {w_acc_x:.2f} | Y: {w_acc_y:.2f}")
    with c3:
        st.metric("Tri-Axial VDV Total", f"{vdv_total:.3f} m/s¹·⁷⁵", f"{vdv_total - prev_vdv:.4f}", delta_color="inverse")
        st.caption("Combined 4th-power shock dose")
    with c4:
        cf_color = "#FF4444" if cf_val > 9.0 else "#00CC66"
        st.metric("Crest Factor (Peak/RMS)", f"{cf_val:.2f}")
        st.markdown(f"<p style='color:{cf_color}; font-weight:bold; font-size:12px;'>{'⚠️ ' + shock_flag if cf_val > 9.0 else '✅ ' + shock_flag}</p>", unsafe_allow_html=True)
    with c5:
        grip_color = "#00CC66" if tire_load >= 0.40 else ("#FFA500" if tire_load >= 0.15 else "#FF3300")
        st.metric("Tire Load Ratio (η_min)", f"{tire_load:.2f}")
        st.markdown(f"<p style='color:{grip_color}; font-weight:bold; font-size:12px;'>{holding_tier}</p>", unsafe_allow_html=True)
    with c6:
        st.metric("Speed / Dominant Freq", f"{latest['avg_speed']:.1f} km/h")
        st.caption(f"Vertical Peak: **{dom_freq:.1f} Hz**")

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
        st.subheader("🌐 Tri-Axial ISO 2631-1 Vibration Breakdown")
        # Multi-trace chart for Tri-axial vibration
        fig_acc = go.Figure()
        if 'total_acc_v' in df_viz.columns:
            fig_acc.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["total_acc_v"], name="Total Vibration (a_v)", line=dict(color="#FFFFFF", width=3)))
        fig_acc.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["weighted_acc_z"], name="Vertical (a_wz, Wk)", line=dict(color="#FF7F0E", width=2)))
        if 'weighted_acc_x' in df_viz.columns:
            fig_acc.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["weighted_acc_x"], name="Longitudinal (a_wx, Wd)", line=dict(color="#00D4FF", width=1.5, dash="dot")))
        if 'weighted_acc_y' in df_viz.columns:
            fig_acc.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["weighted_acc_y"], name="Lateral (a_wy, Wd)", line=dict(color="#FF00AA", width=1.5, dash="dot")))
        
        # Add ISO Threshold Lines
        fig_acc.add_hline(y=0.315, line_dash="dash", line_color="green", annotation_text="Comfortable (<0.315)")
        fig_acc.add_hline(y=0.63, line_dash="dash", line_color="yellow", annotation_text="Mild Uncomfort. (0.63)")
        fig_acc.add_hline(y=1.25, line_dash="dash", line_color="red", annotation_text="Uncomfortable (1.25)")
        
        fig_acc.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_acc, use_container_width=True)

    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.subheader("📈 Tri-Axial Shock Dose (VDV) Trends")
        fig_vib = go.Figure()
        if 'vdv_total' in df_viz.columns:
            fig_vib.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["vdv_total"], name="Total VDV (m/s^1.75)", fill='tozeroy', line=dict(color="#00FFAA", width=2.5)))
        fig_vib.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["weighted_acc_z"], name="Vertical a_wz", line=dict(color="#FF7F0E", width=2)))
        fig_vib.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_vib, use_container_width=True)
        fig_vib.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["weighted_acc_z"], name="Weighted Acc (a_w)", fill='tozeroy', line=dict(color="#FF7F0E")))
        if 'vdv_acc_z' in df_viz.columns:
            fig_vib.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["vdv_acc_z"], name="VDV", line=dict(color="#00FFAA", width=2)))
        fig_vib.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_vib, use_container_width=True)

    with row2_right:
        st.subheader("🛡️ Road-Holding Safety: Dynamic Tire Load (η)")
        fig_tire = go.Figure()
        if 'min_tire_load' in df_viz.columns:
            fig_tire.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["min_tire_load"], name="Min Tire Load Ratio (η_min)", line=dict(color="#00D4FF", width=2.5)))
        if 'avg_tire_load' in df_viz.columns:
            fig_tire.add_trace(go.Scatter(x=df_viz["start_time"], y=df_viz["avg_tire_load"], name="Avg Tire Load (η_avg)", line=dict(color="#AAAAAA", width=1.5, dash="dot")))
            
        # Add Safety Threshold Lines
        fig_tire.add_hline(y=1.0, line_dash="solid", line_color="#555555", annotation_text="Nominal Equilibrium (η = 1.0)")
        fig_tire.add_hline(y=0.40, line_dash="dash", line_color="yellow", annotation_text="Marginal Grip (η = 0.40)")
        fig_tire.add_hline(y=0.15, line_dash="dash", line_color="red", annotation_text="CRITICAL: Wheel Lift Hazard (η = 0.15)")
        fig_tire.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), yaxis_range=[0.0, max(2.5, df_viz['min_tire_load'].max() + 0.5) if 'min_tire_load' in df_viz.columns else 2.5], legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_tire, use_container_width=True)

    # Historical Data Table
    with st.expander("Extended History Data (View All Samples)"):
        st.write(f"Displaying all {len(df_viz)} processed windows for **{selected_vehicle}** - Test: **{selected_test}**")
        cols_to_show = [c for c in ['start_time', 'vehicle_id', 'comfort_score', 'iso_comfort_tier', 'total_acc_v', 'weighted_acc_z', 'weighted_acc_x', 'weighted_acc_y', 'vdv_total', 'min_tire_load', 'road_holding_tier', 'crest_factor_z', 'shock_event', 'dominant_freq_z', 'avg_speed', 'rms_acc_z', 'peak_acc_z'] if c in df_viz.columns]
        st.dataframe(df_viz[cols_to_show].sort_values("start_time", ascending=False), use_container_width=True)

else:
    st.warning(" No data available for the selected flight/vehicle criteria.")

# Footer
st.markdown("---")
st.caption("Vehicle Dynamics & Comfort AI Lab | Real-time Streaming Architecture")
