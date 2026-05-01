"""
app.py — Streamlit Dashboard for Pearls AQI Predictor
"""

import sys
import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

.metric-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-4px); }

.aqi-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 8px 0;
}

.alert-box {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    border-radius: 12px;
    padding: 16px 24px;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.7; }
}

.forecast-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.shap-bar {
    height: 18px;
    border-radius: 9px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    margin: 4px 0;
}

h1, h2, h3 { color: #fff !important; }
.stSelectbox label { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)

# ── API Helper ───────────────────────────────────────────────
API_BASE = f"http://localhost:{config.API_PORT}"

@st.cache_data(ttl=config.CACHE_TTL_SECONDS)
def get_prediction(city: str, shap: bool = True) -> dict | None:
    """Call the FastAPI backend or fall back to direct prediction."""
    try:
        resp = requests.get(f"{API_BASE}/predict", params={"city": city, "shap": shap}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # Direct fallback (no API running)
        try:
            from src.fetch_data import fetch_city_data
            from src.predict import predict_next_days
            weather, aqi, hist = fetch_city_data(city, use_cache=True)
            return predict_next_days(city, weather, aqi, hist, compute_shap=shap)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None

@st.cache_data(ttl=3600)
def get_historical_data(city: str) -> pd.DataFrame:
    try:
        from src.fetch_data import fetch_city_data
        _, _, hist = fetch_city_data(city, use_cache=True)
        return hist
    except Exception:
        return pd.DataFrame()

def get_aqi_color(aqi: float) -> str:
    cat = config.get_aqi_category(aqi)
    return cat["color"]

def aqi_gauge(aqi: float, title: str = "Current AQI") -> go.Figure:
    color = get_aqi_color(aqi)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi,
        title={"text": title, "font": {"size": 18, "color": "white"}},
        number={"font": {"color": color, "size": 48}},
        gauge={
            "axis": {"range": [0, 500], "tickcolor": "white",
                     "tickfont": {"color": "white"}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   50],  "color": "rgba(0,228,0,0.2)"},
                {"range": [51,  100], "color": "rgba(255,255,0,0.2)"},
                {"range": [101, 150], "color": "rgba(255,126,0,0.2)"},
                {"range": [151, 200], "color": "rgba(255,0,0,0.2)"},
                {"range": [201, 300], "color": "rgba(143,63,151,0.2)"},
                {"range": [301, 500], "color": "rgba(126,0,35,0.2)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": aqi},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(t=40, b=10, l=20, r=20),
        height=260,
    )
    return fig

def forecast_chart(daily_preds: list[dict]) -> go.Figure:
    dates = [d["date"] for d in daily_preds]
    means = [d["aqi_mean"] for d in daily_preds]
    maxes = [d["aqi_max"]  for d in daily_preds]
    mins  = [d["aqi_min"]  for d in daily_preds]
    colors= [d["color"]    for d in daily_preds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=maxes + mins[::-1],
        fill="toself",
        fillcolor="rgba(102,126,234,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Min–Max Range",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=means,
        mode="lines+markers+text",
        text=[f"{v:.0f}" for v in means],
        textposition="top center",
        textfont=dict(color="white", size=14),
        line=dict(color="#667eea", width=3),
        marker=dict(size=12, color=colors, line=dict(color="white", width=2)),
        name="Daily Mean AQI",
    ))
    fig.add_hline(y=150, line_dash="dash", line_color="#ff4b2b",
                  annotation_text="⚠️ Unhealthy", annotation_font_color="#ff4b2b")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(showgrid=False, color="white"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                   color="white", range=[0, max(maxes)*1.2]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        margin=dict(t=20, b=30, l=40, r=20),
        height=300,
    )
    return fig

def pollutant_radar(pollutants: dict) -> go.Figure:
    cats   = list(pollutants.keys())
    values = [round(v, 1) for v in pollutants.values()]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(102,126,234,0.3)",
        line=dict(color="#667eea", width=2),
        marker=dict(size=8, color="#764ba2"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, color="rgba(255,255,255,0.4)"),
            angularaxis=dict(color="white"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(t=30, b=30, l=30, r=30),
        height=280,
        showlegend=False,
    )
    return fig

def historical_chart(hist_df: pd.DataFrame) -> go.Figure:
    if hist_df.empty:
        return go.Figure()
    df = hist_df.tail(168)  # Last 7 days (hourly)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"],
        mode="lines",
        line=dict(color="#667eea", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(102,126,234,0.15)",
        name="AQI",
    ))
    for threshold, label, color in [(50,"Good","#00e400"),(100,"Moderate","#ffff00"),(150,"Sensitive","#ff7e00")]:
        fig.add_hline(y=threshold, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_font_color=color)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(showgrid=False, color="white"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", color="white"),
        margin=dict(t=10, b=30, l=40, r=20),
        height=250,
        showlegend=False,
    )
    return fig

def shap_chart(top_features: list[dict]) -> go.Figure:
    if not top_features:
        return go.Figure()
    feats = [f["feature"] for f in top_features[:10]][::-1]
    vals  = [f["importance"] for f in top_features[:10]][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=feats,
        orientation="h",
        marker=dict(
            color=vals,
            colorscale="Viridis",
            showscale=False,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(title="Mean |SHAP|", color="white", showgrid=True,
                   gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(color="white"),
        margin=dict(t=10, b=30, l=10, r=20),
        height=300,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 24px 0 8px 0;">
  <h1 style="font-size:2.8rem; font-weight:900; 
     background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     margin-bottom:4px;">
    🌬️ Pearls AQI Predictor
  </h1>
  <p style="color:#aaa; font-size:1.05rem;">
    Real-time Air Quality Index forecasting powered by Machine Learning
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    city = st.selectbox(
        "📍 Select City",
        options=config.SUPPORTED_CITIES,
        index=0,
        key="city_selector",
    )



    st.markdown("---")
    include_shap = st.toggle("🔍 Show SHAP Explainability", value=True)
    auto_refresh  = st.toggle("🔄 Auto Refresh (5 min)", value=False)

    st.markdown("---")

    if st.button("🚀 Get Prediction", type="primary", use_container_width=True):
        st.cache_data.clear()

    st.markdown("---")
    st.markdown("### 📊 AQI Scale")
    aqi_levels = [
        ("😊 Good",              "0–50",   "#00e400"),
        ("😐 Moderate",          "51–100",  "#ffff00"),
        ("😷 Unhealthy (Sens.)", "101–150", "#ff7e00"),
        ("🤢 Unhealthy",         "151–200", "#ff0000"),
        ("🚨 Very Unhealthy",    "201–300", "#8f3f97"),
        ("☠️ Hazardous",         "301–500", "#7e0023"),
    ]
    for label, rng, color in aqi_levels:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<div style="width:12px;height:12px;border-radius:50%;background:{color};flex-shrink:0;"></div>'
            f'<span style="color:#ddd;font-size:0.85rem;">{label} <span style="color:#888">({rng})</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Auto-refresh logic ────────────────────────────────────────
if auto_refresh:
    time.sleep(300)
    st.rerun()

# ── Fetch data ────────────────────────────────────────────────
with st.spinner(f"🌐 Fetching data for **{city}** …"):
    data = get_prediction(city, shap=include_shap)
    hist_df = get_historical_data(city)

if not data:
    st.error("❌ Could not fetch prediction. Make sure the API is running or you have a trained model.")
    st.stop()

# ── Alert banner ──────────────────────────────────────────────
if data.get("current_alert"):
    st.markdown(
        f'<div class="alert-box">🚨 AIR QUALITY ALERT — '
        f'{city} current AQI is {data["current_aqi"]} ({data["current_category"]}). '
        f'Limit outdoor activities!</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

# Check any forecast day alert
for day in data.get("daily_predictions", []):
    if day.get("alert"):
        st.warning(
            f"⚠️ **Day {day['day']} ({day['date']})** forecast AQI {day['aqi_mean']} — "
            f"{day['category']}. Unhealthy air quality predicted!"
        )

# ── Row 1: Current conditions ─────────────────────────────────
col_gauge, col_weather, col_pollutants = st.columns([1.2, 1, 1])

with col_gauge:
    st.markdown(f"### 📍 {city} — Current AQI")
    st.plotly_chart(aqi_gauge(data["current_aqi"]), use_container_width=True, key="gauge")
    cat   = data["current_category"]
    color = data["current_color"]
    emoji = data["current_emoji"]
    st.markdown(
        f'<div style="text-align:center;">'
        f'<span class="aqi-badge" style="background:{color}20;color:{color};'
        f'border:2px solid {color};">{emoji} {cat}</span></div>',
        unsafe_allow_html=True,
    )

with col_weather:
    st.markdown("### 🌡️ Weather Conditions")
    w = data.get("current_weather", {})
    weather_items = [
        ("🌡️ Temperature", f"{w.get('temp', 'N/A')} °C"),
        ("💧 Humidity",    f"{w.get('humidity', 'N/A')} %"),
        ("💨 Wind Speed",  f"{w.get('wind_speed', 'N/A')} m/s"),
        ("🔵 Pressure",    f"{w.get('pressure', 'N/A')} hPa"),
    ]
    for icon_label, val in weather_items:
        st.markdown(
            f'<div class="metric-card" style="margin-bottom:10px;">'
            f'<div style="color:#aaa;font-size:0.85rem;">{icon_label}</div>'
            f'<div style="color:#fff;font-size:1.4rem;font-weight:700;">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_pollutants:
    st.markdown("### 🧪 Pollutants")
    p = data.get("current_pollutants", {})
    if p:
        st.plotly_chart(pollutant_radar(p), use_container_width=True, key="radar")

st.divider()

# ── Row 2: 3-Day Forecast ─────────────────────────────────────
st.markdown("## 📅 3-Day AQI Forecast")
col_f1, col_f2, col_f3 = st.columns(3)
forecast_cols = [col_f1, col_f2, col_f3]
days          = data.get("daily_predictions", [])

for i, (col, day) in enumerate(zip(forecast_cols, days)):
    with col:
        alert_badge = "⚠️ ALERT" if day["alert"] else ""
        st.markdown(
            f'<div class="forecast-card">'
            f'<div style="color:#aaa;font-size:0.8rem;">DAY {day["day"]}</div>'
            f'<div style="color:#fff;font-weight:700;font-size:1rem;">{day["date"]}</div>'
            f'<div style="font-size:3rem;margin:8px 0;">{day["emoji"]}</div>'
            f'<div style="color:{day["color"]};font-size:2.4rem;font-weight:900;">{day["aqi_mean"]:.0f}</div>'
            f'<div style="color:#aaa;font-size:0.8rem;">AQI (Mean)</div>'
            f'<div style="color:{day["color"]};font-size:1rem;font-weight:600;margin:4px 0;">{day["category"]}</div>'
            f'<div style="color:#888;font-size:0.78rem;">↑ {day["aqi_max"]:.0f} &nbsp;|&nbsp; ↓ {day["aqi_min"]:.0f}</div>'
            f'<div style="color:#ff4b2b;font-size:0.85rem;margin-top:6px;">{alert_badge}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📈 Forecast Trend")
st.plotly_chart(forecast_chart(days), use_container_width=True, key="forecast_chart")

st.divider()

# ── Row 3: Historical + SHAP ──────────────────────────────────
col_hist, col_shap = st.columns(2)

with col_hist:
    st.markdown("### 📜 Historical AQI (Last 7 Days)")
    if not hist_df.empty and "aqi" in hist_df.columns and "timestamp" in hist_df.columns:
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
        st.plotly_chart(historical_chart(hist_df), use_container_width=True, key="hist_chart")
    else:
        st.info("No historical data available yet. Run the data pipeline first.")

with col_shap:
    st.markdown("### 🔍 SHAP Feature Importance")
    if include_shap:
        shap_data = data.get("shap_explanation", {})
        top_feats  = shap_data.get("top_features", [])
        if top_feats:
            st.plotly_chart(shap_chart(top_feats), use_container_width=True, key="shap_chart")
            st.caption("Higher SHAP values = stronger influence on AQI prediction")
        else:
            st.info(f"SHAP not available: {shap_data.get('error', 'unknown reason')}")
    else:
        st.info("Enable SHAP in sidebar to see feature importance.")

st.divider()

# ── Row 4: Model Info ─────────────────────────────────────────
st.markdown("### 🤖 Deployed Model Details")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
metrics = data.get("model_metrics", {})
model_cards = [
    ("Currently Active", data.get("model_name", "N/A"), "🏆"),
    ("MAE",   f"{metrics.get('mae', 'N/A')}",  "📉"),
    ("RMSE",  f"{metrics.get('rmse', 'N/A')}", "📊"),
    ("R²",    f"{metrics.get('r2', 'N/A')}",   "🎯"),
]
for col, (label, val, icon) in zip([col_m1, col_m2, col_m3, col_m4], model_cards):
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:1.8rem;">{icon}</div>'
            f'<div style="color:#aaa;font-size:0.85rem;">{label}</div>'
            f'<div style="color:#667eea;font-size:1.2rem;font-weight:700;">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🥇 Training Leaderboard (All Evaluated Models)")
st.caption("During the last daily training run, the pipeline automatically tested the following models. The model with the lowest RMSE was automatically deployed.")

try:
    import pandas as pd
    comp_df = pd.read_csv("models/model_comparison.csv")
    
    # Highlight the deployed model
    deployed_name = data.get("model_name", "")
    
    def highlight_deployed(row):
        if row['model_name'] == deployed_name:
            return ['background-color: rgba(102, 126, 234, 0.3)'] * len(row)
        return [''] * len(row)
        
    st.dataframe(
        comp_df.style.apply(highlight_deployed, axis=1).format({
            "rmse": "{:.4f}",
            "mae": "{:.4f}",
            "r2": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )
except Exception as e:
    st.info("Leaderboard data not currently available.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#555;font-size:0.8rem;">'
    'Pearls AQI Predictor © 2026 &nbsp;|&nbsp; '
    'Data: OpenWeatherMap API &nbsp;|&nbsp; '
    f'Last updated: {data.get("generated_at", "")[:16]}'
    '</div>',
    unsafe_allow_html=True,
)
