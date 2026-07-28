"""
Repo_9_Energy_Consumption — Home.py
Author : Mohamed · M3
"""
# streamlit run "E:\FINAL PROJECTS\P9_household_power_consumption\Home.py"

import pathlib
import streamlit as st

st.set_page_config(page_title="Energy Consumption · M3", page_icon="⚡", layout="wide")
LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### ⚡ Energy Consumption")
    st.markdown("M3 · ML Engine · P9")
    st.divider()
    st.markdown("**Navigate:**")
    st.markdown("📊 EDA Dashboard → 13 tabs")
    st.markdown("🤖 ML Models     → 5 tabs")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f4f7fb;}
.hero{background:linear-gradient(135deg,#1a237e,#f57f17);
      padding:48px 40px;border-radius:14px;margin-bottom:28px;}
.hero h1{color:#ffffff !important;font-size:2.4rem;font-weight:800;margin:0 0 8px 0;}
.hero p{color:#fff9c4 !important;font-size:1.08rem;margin:0;}
.card{background:#ffffff;border-radius:10px;padding:22px 24px;
      box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:4px solid #f57f17;}
.card h3{color:#e65100 !important;margin:0 0 8px 0;font-size:1.05rem;}
.card p{color:#37474f !important;font-size:0.92rem;margin:0;line-height:1.6;}
.stat-card{background:#ffffff;border-radius:10px;padding:18px;text-align:center;
           box-shadow:0 2px 10px rgba(0,0,0,0.07);border-bottom:3px solid #f57f17;}
.stat-num{font-size:1.9rem;font-weight:800;color:#e65100 !important;}
.stat-lbl{font-size:0.82rem;color:#546e7a !important;margin-top:4px;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>⚡ Household Energy Consumption</h1>
  <p>End-to-end ML pipeline · ~1M minute readings · UCI Dataset · M3 Portfolio · Project 9 of 12</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
for col, (num, lbl) in zip([c1,c2,c3,c4,c5],[
    ("~17K","Hourly Records"), ("2 Years","Dec 2006–2008"),
    ("9","Raw Features"), ("13","EDA Tabs"), ("12","ML Models")]):
    col.markdown(f"""<div class="stat-card">
      <div class="stat-num">{num}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📌 About This Project")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="card"><h3>🎯 Objective</h3>
    <p>Analyse household electricity consumption patterns, detect anomalies,
    decompose time series, and predict high-consumption periods using ML.</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card"><h3>📊 Dataset</h3>
    <p>UCI Household Power Consumption · ~1M minute readings resampled to hourly.
    Engineered: Hour, Season, Period, is_peak, Unmetered_power,
    Rolling_24h, Rolling_7d.</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card"><h3>🔑 Key Signals</h3>
    <p>Hour of day · Season · Peak hours (7–9 AM, 6–9 PM) ·
    Sub-metering zones (kitchen, laundry, HVAC) ·
    7-day rolling average · Unmetered power.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    st.markdown("### 📈 EDA Dashboard — 13 Tabs")
    for num, name, desc in [
        ("1","Data Overview","Shape, types, stats, dictionary"),
        ("2","Power Distribution","Global active power histogram + box plot"),
        ("3","Time Series ★","Full timeline + 7-day rolling average"),
        ("4","Hourly Patterns ★","Avg consumption by hour · heatmap hour×day"),
        ("5","Seasonal Analysis ★","Monthly + seasonal consumption patterns"),
        ("6","Sub-metering ★","Kitchen vs laundry vs HVAC breakdown"),
        ("7","STL Decomposition ★","Trend + Seasonal + Residual"),
        ("8","Anomaly Detection ★","Z-score spikes + IQR outliers on timeline"),
        ("9","Feature Engineering","Rolling features + peak flag + unmetered power"),
        ("10","Correlation","Heatmap + top power predictors"),
        ("11","A/B Test ★","Peak hours vs off-peak — Welch T-test + Cohen's d"),
        ("12","Multicollinearity","VIF analysis"),
        ("13","Insights & Report","Findings + recommendations + download"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

with col5:
    st.markdown("### 🤖 ML Models — 5 Tabs")
    for num, name, desc in [
        ("1","Model Training","6 Reg + 6 Clf · individual buttons"),
        ("2","Regression Results","R², MAE, RMSE · predict Global_active_power"),
        ("3","Classification Results","F1, Accuracy, ROC-AUC · predict high_consumption"),
        ("4","Feature Importance","Top energy predictors"),
        ("5","Predict","Interactive consumption forecasting"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("**Target balance: ~50/50**\n\nNo class_weight adjustment needed.\n\n"
            "Evaluate regression with **R², MAE, RMSE**.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#90a4ae;font-size:0.85rem;'>"
            "Mohamed · M3 · ML Engine Portfolio · Project 9 of 12 · Energy Consumption</p>",
            unsafe_allow_html=True)
