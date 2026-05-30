"""
dashboard/app.py
================
Streamlit real-time analytics dashboard for the Conveyor Belt CV System.

Reads live data from the SQLite database written by the main application and
presents it in an industrial monitoring interface.

Launch with:
    cd /path/to/source_code
    streamlit run dashboard/app.py -- --db data/conveyor.db

Or simply:
    streamlit run dashboard/app.py

Features:
  - Auto-refreshes every 2 seconds
  - Big metric cards: Total, Normal, Defective, Defect Rate
  - Throughput-per-minute bar chart (Plotly)
  - Defect rate trend line chart
  - Scrollable alert log table
  - Sidebar: controls, DB path selection, refresh interval
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so we can import src.*
# ---------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).parent
ROOT = DASHBOARD_DIR.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.database import Database


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Conveyor Belt Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark industrial theme */
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3a);
        border: 1px solid #2d3550;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 6px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8899bb;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .alert-row-defect { color: #ff6b6b; }
    .alert-row-normal { color: #6bffb1; }
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar() -> tuple[str, int]:
    """Render sidebar and return (db_path, refresh_interval_ms)."""
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/conveyor.png",
        width=72,
    )
    st.sidebar.title("🏭 Conveyor Monitor")
    st.sidebar.markdown("---")

    db_path = st.sidebar.text_input(
        "SQLite Database Path",
        value=str(ROOT / "data" / "conveyor.db"),
        help="Path to the SQLite file written by app.py",
    )

    refresh_sec = st.sidebar.slider(
        "Auto-refresh interval (s)", min_value=1, max_value=30, value=2
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Pipeline:**\n"
        "- YOLO v8/v11 Detection\n"
        "- ByteTrack Tracking\n"
        "- Virtual Counting Zones\n"
        "- MobileNetV3 Defect Classifier\n"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Group: Do Duy Loi · Trinh Chan Duy · Dang Vo Hong Phuc\n\n"
        "University of Science — 2026"
    )

    return db_path, refresh_sec * 1000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_db(db_path: str) -> Database:
    return Database(db_path)


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------

def metric_card(col, label: str, value, colour: str, icon: str = ""):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{colour}">{icon} {value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main():
    db_path, refresh_ms = sidebar()

    st.title("🏭 Conveyor Belt Monitoring Dashboard")
    st.caption(
        f"Live data from `{db_path}` · Auto-refreshes every {refresh_ms // 1000}s"
    )

    # ---- Check if DB exists ----
    if not Path(db_path).exists():
        st.warning(
            "⚠️ Database not found. Start `app.py` first to generate data.\n\n"
            f"Expected at: `{db_path}`"
        )
        st.stop()

    db = get_db(db_path)

    # ---- Fetch data ----
    stats         = db.get_stats_summary()
    throughput    = db.get_throughput_by_minute(60)
    recent_alerts = db.get_recent_alerts(100)
    recent_prods  = db.get_recent_products(200)

    total     = stats.get("total",     0)
    normal    = stats.get("normal",    0)
    defective = stats.get("defective", 0)
    pending   = stats.get("pending",   0)
    defect_rate = (defective / max(total, 1)) * 100

    # ---- Metric cards ----
    st.markdown("### 📊 Live Statistics")
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total Products",  total,      "#64b5f6", "📦")
    metric_card(c2, "Normal",          normal,     "#81c784", "✅")
    metric_card(c3, "Defective",       defective,  "#e57373", "❌")
    metric_card(c4, "Defect Rate",     f"{defect_rate:.1f}%", "#ffd54f", "📉")

    st.markdown("---")

    # ---- Charts ----
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📈 Throughput per Minute")
        if throughput:
            df_tp = pd.DataFrame(throughput)
            fig = px.bar(
                df_tp,
                x="minute",
                y="count",
                color="count",
                color_continuous_scale="Blues",
                labels={"minute": "Minute", "count": "Products"},
            )
            fig.update_layout(
                plot_bgcolor="#1a1f2e",
                paper_bgcolor="#1a1f2e",
                font_color="#c0cfe8",
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=40),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No throughput data yet. Products will appear as the pipeline runs.")

    with col_right:
        st.markdown("### 📉 Defect Rate Trend")
        if recent_prods:
            df_p = pd.DataFrame(recent_prods)
            df_p["timestamp"] = pd.to_datetime(df_p["timestamp"], unit="s")
            df_p = df_p.sort_values("timestamp")
            # Rolling defect rate (every 10 products)
            df_p["is_defective"] = (df_p["status"] == "Defective").astype(int)
            df_p["rolling_rate"] = (
                df_p["is_defective"].rolling(10, min_periods=1).mean() * 100
            )
            fig2 = px.line(
                df_p,
                x="timestamp",
                y="rolling_rate",
                labels={"timestamp": "Time", "rolling_rate": "Defect Rate (%)"},
                line_shape="spline",
                color_discrete_sequence=["#e57373"],
            )
            fig2.add_hline(y=15, line_dash="dot", line_color="#ffd54f",
                           annotation_text="15% threshold")
            fig2.update_layout(
                plot_bgcolor="#1a1f2e",
                paper_bgcolor="#1a1f2e",
                font_color="#c0cfe8",
                margin=dict(l=10, r=10, t=10, b=40),
                yaxis=dict(range=[0, 100]),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No product classification data yet.")

    st.markdown("---")

    # ---- Alert log ----
    st.markdown("### 🚨 Alert Log")
    if recent_alerts:
        df_alerts = pd.DataFrame(recent_alerts)
        df_alerts["timestamp"] = pd.to_datetime(df_alerts["timestamp"], unit="s")
        df_alerts = df_alerts[["timestamp", "track_id", "alert_type", "message"]].copy()
        df_alerts.columns = ["Time", "Track ID", "Alert Type", "Message"]
        df_alerts = df_alerts.sort_values("Time", ascending=False)
        st.dataframe(df_alerts, use_container_width=True, height=250)
    else:
        st.success("✅ No alerts — all products within normal parameters.")

    # ---- Recent products table ----
    with st.expander("📋 Recent Products Log", expanded=False):
        if recent_prods:
            df_p2 = pd.DataFrame(recent_prods)
            df_p2["timestamp"] = pd.to_datetime(df_p2["timestamp"], unit="s")
            df_p2 = df_p2[["id", "track_id", "timestamp", "status", "confidence", "frame_idx"]].copy()
            df_p2.columns = ["DB ID", "Track ID", "Time", "Status", "Confidence", "Frame"]
            df_p2 = df_p2.sort_values("DB ID", ascending=False)
            st.dataframe(df_p2, use_container_width=True, height=300)

    # ---- Footer / auto-refresh ----
    st.markdown("---")
    placeholder = st.empty()
    with placeholder.container():
        st.caption(f"🔄 Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
                   f"Total DB records: {total}")

    # Streamlit auto-rerun
    time.sleep(refresh_ms / 1000.0)
    st.rerun()


if __name__ == "__main__":
    main()
