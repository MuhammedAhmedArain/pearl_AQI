"""
app.py — Premium Dashboard for Pearls AQI Predictor
"""

import sys
import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

.stApp {
    background-color: #1e293b;
    color: #f8fafc;
}

/* Hide Streamlit Default Header/Footer/Deploy Button */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stToolbar"] {visibility: hidden;}

/* Custom Gradients and Typography */
.title-text {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
    line-height: 1.2;
}

.subtitle-text {
    color: #94a3b8;
    font-size: 1.15rem;
    font-weight: 400;
    margin-top: 0px;
    margin-bottom: 30px;
    letter-spacing: 0.5px;
}

/* AQI Cards */
.aqi-card {
    border-radius: 24px;
    padding: 28px 20px;
    box-shadow: 0 10px 30px -5px rgba(0,0,0,0.5);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.aqi-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.2);
}

.bg-good { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
.bg-moderate { background: linear-gradient(135deg, #d97706 0%, #fbbf24 100%); }
.bg-sensitive { background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); }
.bg-unhealthy { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); }
.bg-very { background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%); }
.bg-haz { background: linear-gradient(135deg, #9f1239 0%, #e11d48 100%); }

.card-title { font-size: 1.1rem; font-weight: 700; opacity: 0.95; text-transform: uppercase; letter-spacing: 1.5px; }
.card-value { font-size: 4.5rem; font-weight: 800; line-height: 1.1; margin: 12px 0; text-shadow: 0 4px 10px rgba(0,0,0,0.2); }
.card-subtitle { font-size: 1.2rem; font-weight: 600; opacity: 0.95; margin-bottom: 4px; }
.card-details { font-size: 0.9rem; opacity: 0.8; }

/* Alert Boxes */
.alert-danger {
    background: linear-gradient(135deg, #7f1d1d, #9f1239);
    border-radius: 16px; padding: 20px; color: white; font-weight: 600;
    box-shadow: 0 4px 20px rgba(159, 18, 57, 0.4);
    animation: pulse 2s infinite; border-left: 8px solid #f43f5e;
    font-size: 1.1rem;
}
.alert-warning {
    background: linear-gradient(135deg, #9a3412, #c2410c);
    border-radius: 16px; padding: 20px; color: white; font-weight: 600;
    box-shadow: 0 4px 20px rgba(194, 65, 12, 0.4); 
    border-left: 8px solid #f97316; font-size: 1.1rem;
}

@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.8; } }

/* Glass Model Card */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}

.model-tag {
    display: inline-block; background: rgba(56, 189, 248, 0.15);
    color: #38bdf8; padding: 6px 16px; border-radius: 30px; 
    font-weight: 700; font-size: 0.95rem; border: 1px solid rgba(56, 189, 248, 0.3);
}

/* Base Streamlit overrides */
.stSelectbox > div > div { background-color: rgba(30, 41, 59, 0.8) !important; border-radius: 12px; color: white; }
.stButton > button { border-radius: 12px !important; height: 46px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── API & Data Helpers ─────────────────────────────────────────

API_BASE = f"http://localhost:{config.API_PORT}"

@st.cache_data(ttl=config.CACHE_TTL_SECONDS)
def get_prediction(city: str, shap: bool = True) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}/predict", params={"city": city, "shap": shap}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception:
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

def get_aqi_category_class(aqi: float) -> str:
    if aqi <= 50: return "bg-good"
    elif aqi <= 100: return "bg-moderate"
    elif aqi <= 150: return "bg-sensitive"
    elif aqi <= 200: return "bg-unhealthy"
    elif aqi <= 300: return "bg-very"
    else: return "bg-haz"

# ── Plotly Charts ──────────────────────────────────────────────

def combined_trend_chart(hist_df: pd.DataFrame, daily_preds: list[dict], current_aqi: float) -> go.Figure:
    fig = go.Figure()
    
    # 1. Historical Data
    if not hist_df.empty and "timestamp" in hist_df.columns:
        df = hist_df.tail(48) # last 48 hours for better visual scale
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df["timestamp"]), y=df["aqi"],
            mode="lines",
            line=dict(color="#94a3b8", width=2, shape="spline"),
            name="Historical AQI",
            fill="tozeroy",
            fillcolor="rgba(148, 163, 184, 0.08)",
            hovertemplate="<b>Date</b>: %{x}<br><b>AQI</b>: %{y:.0f}<extra></extra>"
        ))
        last_date = pd.to_datetime(df["timestamp"]).iloc[-1]
    else:
        last_date = pd.Timestamp.now()
        
    # 2. Predicted Data
    pred_dates = [last_date] + [pd.to_datetime(d["date"]) for d in daily_preds]
    pred_vals = [current_aqi] + [d["aqi_mean"] for d in daily_preds]
    
    fig.add_trace(go.Scatter(
        x=pred_dates, y=pred_vals,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=4, shape="spline", dash="dot"),
        marker=dict(size=12, color="#818cf8", line=dict(color="#0f172a", width=3)),
        name="Predicted Average",
        hovertemplate="<b>Date</b>: %{x|%b %d}<br><b>Predicted AQI</b>: %{y:.0f}<extra></extra>"
    ))
    
    # Thresholds
    fig.add_hline(y=150, line_dash="dash", line_color="#f97316", opacity=0.4, annotation_text="Unhealthy (Sensitive)", annotation_font_color="#f97316")
    fig.add_hline(y=200, line_dash="dash", line_color="#ef4444", opacity=0.4, annotation_text="Unhealthy", annotation_font_color="#ef4444")
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
        xaxis=dict(showgrid=False, title="", tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="AQI Level"),
        margin=dict(t=20, b=20, l=40, r=20), height=380,
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    return fig

def modern_shap_chart(top_features: list[dict]) -> go.Figure:
    if not top_features: return go.Figure()
    feats = [f["feature"].replace("_", " ").title() for f in top_features[:7]][::-1]
    vals  = [f["importance"] for f in top_features[:7]][::-1]
    
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(
            color=vals,
            colorscale="Purp",
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Impact on Prediction"),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#e2e8f0")),
        margin=dict(t=10, b=30, l=10, r=20), height=380,
    )
    return fig

# ── UI Layout Assembly ─────────────────────────────────────────

# Header Title
st.markdown("<div class='title-text'>🌍 AQI Forecast Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Real-time Air Quality Monitoring & 3-Day Prediction Powered by Machine Learning</div>", unsafe_allow_html=True)

# Controls
col_city, col_btn, col_space = st.columns([2, 1, 6])
with col_city:
    city = st.selectbox("Location", options=config.SUPPORTED_CITIES, label_visibility="collapsed")
with col_btn:
    if st.button("🔄 Sync with Cloud", use_container_width=True, help="Force refresh data and reload latest model from Registry"):
        st.cache_data.clear()
        st.success("Cache cleared! Reloading...")
        time.sleep(0.5)
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Fetch Data
with st.spinner("Analyzing atmospheric data models..."):
    data = get_prediction(city, shap=True)
    hist_df = get_historical_data(city)

if not data:
    st.error("❌ Analytics engine offline. Could not fetch data.")
    st.stop()

# SECTION 3: AQI CATEGORY + ALERT
c_aqi = data["current_aqi"]
if c_aqi > 200:
    st.markdown(f"<div class='alert-danger'>🚨 DANGER ALERT: Current AQI is {c_aqi:.0f} ({data['current_category']}). Air quality is extremely hazardous! Limit outdoor exposure immediately.</div><br>", unsafe_allow_html=True)
elif c_aqi > 150:
    st.markdown(f"<div class='alert-warning'>⚠️ WARNING ALERT: Current AQI is {c_aqi:.0f} ({data['current_category']}). Unhealthy air quality. Sensitive groups should wear masks.</div><br>", unsafe_allow_html=True)

# SECTION 1: SUMMARY CARDS
st.markdown("<h3 style='color: white; margin-bottom: 20px;'>📊 AQI Overview</h3>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

def render_card(col, title, aqi, date, emoji, cat):
    bg = get_aqi_category_class(aqi)
    with col:
        st.markdown(f"""
        <div class="aqi-card {bg}">
            <div class="card-title">{title}</div>
            <div class="card-value">{aqi:.0f}</div>
            <div class="card-subtitle">{emoji} {cat}</div>
            <div class="card-details">{date}</div>
        </div>
        """, unsafe_allow_html=True)

render_card(c1, "CURRENT", data["current_aqi"], "Live Now", data["current_emoji"], data["current_category"])

days = data.get("daily_predictions", [])
if len(days) >= 3:
    render_card(c2, "TOMORROW", days[0]["aqi_mean"], days[0]["date"], days[0]["emoji"], days[0]["category"])
    render_card(c3, "DAY 2", days[1]["aqi_mean"], days[1]["date"], days[1]["emoji"], days[1]["category"])
    render_card(c4, "DAY 3", days[2]["aqi_mean"], days[2]["date"], days[2]["emoji"], days[2]["category"])

st.markdown("<br><br>", unsafe_allow_html=True)

# SECTION 2 & 5: CHARTS (Trend + SHAP)
col_trend, col_shap = st.columns([6, 4], gap="large")

with col_trend:
    st.markdown("<h3 style='color: white;'>📈 AQI Trend & Forecast Analysis</h3>", unsafe_allow_html=True)
    st.plotly_chart(combined_trend_chart(hist_df, days, data["current_aqi"]), use_container_width=True, config={'displayModeBar': False})

with col_shap:
    st.markdown("<h3 style='color: white;'>🔍 Primary Risk Factors (SHAP)</h3>", unsafe_allow_html=True)
    if data.get("shap_explanation", {}).get("top_features"):
        st.plotly_chart(modern_shap_chart(data["shap_explanation"]["top_features"]), use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Factor analysis unavailable.")

st.markdown("<br><br>", unsafe_allow_html=True)

# SECTION 4: MODEL INFO & EXPANDABLE DETAILS
col_model, col_exp = st.columns([4, 6], gap="large")

with col_model:
    st.markdown("<h3 style='color: white;'>🤖 Intelligence Engine</h3>", unsafe_allow_html=True)
    metrics = data.get("model_metrics", {})
    st.markdown(f"""
    <div class="glass-card">
        <span class="model-tag">🧠 {data.get("model_name", "AI Model")}</span>
        <div style="margin-top: 15px; font-size: 0.85rem; color: #94a3b8;">
            📅 Last Trained: <b>{data.get("model_metrics", {}).get("trained_at", "N/A")}</b>
        </div>
        <div style="margin-top: 15px; display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <div style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 4px;">Root Mean Square Error</div>
                <div style="font-size: 2.5rem; font-weight: 800; color: #f8fafc; line-height: 1;">{metrics.get('rmse', '0.0')}</div>
            </div>
            <div style="text-align: right;">
                <div style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 4px;">R² Score</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8; line-height: 1;">{metrics.get('r2', '0.0')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_exp:
    st.markdown("<h3 style='color: white;'>💡 Advanced Analytics</h3>", unsafe_allow_html=True)
    
    with st.expander("🌤️ Show Detailed Atmospheric Conditions", expanded=False):
        w = data.get("current_weather", {})
        c_cols = st.columns(4)
        c_cols[0].metric("Temperature", f"{w.get('temp', 0)} °C")
        c_cols[1].metric("Humidity", f"{w.get('humidity', 0)} %")
        c_cols[2].metric("Wind Speed", f"{w.get('wind_speed', 0)} m/s")
        c_cols[3].metric("Pressure", f"{w.get('pressure', 0)} hPa")
        
    with st.expander("🏆 Show Model Training Leaderboard", expanded=False):
        try:
            comp_df = pd.read_csv("models/model_comparison.csv")
            
            # Simple highlighter for deployed model
            deployed = data.get("model_name", "")
            def highlight(row):
                return ['background-color: rgba(56, 189, 248, 0.15)'] * len(row) if row['model_name'] == deployed else [''] * len(row)
                
            st.dataframe(
                comp_df.style.apply(highlight, axis=1).format({"rmse": "{:.4f}", "mae": "{:.4f}", "r2": "{:.4f}"}),
                use_container_width=True, hide_index=True
            )
        except Exception:
            st.info("Leaderboard data unavailable locally.")

# Footer
st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center;color:#64748b;font-size:0.85rem;padding:20px 0;">'
    f'Pearls AQI Predictor © 2026 &nbsp;&bull;&nbsp; '
    f'Data: OpenWeatherMap API &nbsp;&bull;&nbsp; '
    f'Last updated: {data.get("generated_at", "")[:16]}'
    f'</div>',
    unsafe_allow_html=True
)
