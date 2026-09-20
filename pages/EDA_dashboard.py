"""
Repo_9_Energy_Consumption — EDA_dashboard.py  (13 Tabs)
Author : Mohamed · M3
Dataset: UCI Household Power Consumption · hourly resampled
"""
# streamlit run "E:\FINAL PROJECTS\P9_household_power_consumption\EDA_dashboard.py"
import streamlit as st

import pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore


warnings.filterwarnings("ignore")
S = st.session_state

st.set_page_config(page_title="EDA · Energy Consumption · M3",
                   page_icon="⚡", layout="wide")

LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"
DATA = pathlib.Path(__file__).parent.parent / "data" / "energy_clean.csv"
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### ⚡ EDA Dashboard")
    st.markdown("Energy Consumption · 13 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    _uploaded = st.file_uploader("Upload Clean CSV", type=["csv"],
                                  key="p9_eda_upload")
    # Capture bytes IMMEDIATELY — before any other widget reads the object
    if _uploaded is not None:
        S["p9_file_bytes"] = _uploaded.read()
        _uploaded.seek(0)
        st.success(f"✅ {_uploaded.name}")
    elif DATA.exists():
        st.success("✅ Using: energy_clean.csv (local)")
    else:
        st.warning("⚠️ Upload energy_clean.csv")

CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","light":"#e3f2fd","dark":"#1a237e",
       "purple":"#6a1b9a","amber":"#f57f17","cyan":"#00838f","grey":"#546e7a"}

#=============================================================================

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{background:#1a2633;border:1.5px dashed #4a7fa5;border-radius:8px;padding:6px;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] *{color:#e0e8f0 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:#1a2633 !important;border:none !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *{color:#a0bcd4 !important;}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:#1565c0 !important;color:#ffffff !important;border:none !important;border-radius:6px !important;}
.main{background:#f4f7fb;}
div[data-testid="metric-container"]{background:#fff8e1;border-left:4px solid #f57f17;border-radius:6px;padding:10px 14px;}
.sec-header{background:linear-gradient(90deg,#e65100,#1565c0);color:#ffffff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}
.insight-box{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.insight-box p{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}
.warn-box{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.warn-box p{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}
.info-box{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.info-box p{color:#0d2a4a !important;margin:0;font-size:0.93rem;line-height:1.6;}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)

# ── LOAD ─────────────────────────────────────────────────────
def load_data(file_bytes: bytes = None) -> pd.DataFrame:
    import io as _io
    if file_bytes is not None:
        df = pd.read_csv(_io.BytesIO(file_bytes), sep=",", decimal=".")
    else:
        df = pd.read_csv(DATA, sep=",", decimal=".")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    return df

# ── LOAD: use session_state bytes (captured in sidebar) ──────
import io as _io_main

_stored_bytes = S.get("p9_file_bytes", None)

if _stored_bytes:
    # Bytes captured from upload inside sidebar — always reliable
    _df_raw = pd.read_csv(_io_main.BytesIO(_stored_bytes), sep=",", decimal=".")
    _df_raw = _df_raw.loc[:, ~_df_raw.columns.str.startswith("Unnamed")]
    _df_raw.columns = _df_raw.columns.str.strip()
    if "DateTime" in _df_raw.columns:
        _df_raw["DateTime"] = pd.to_datetime(_df_raw["DateTime"], errors="coerce")
        _df_raw = _df_raw.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    df = _df_raw
elif DATA.exists():
    # No upload → load directly from data/ folder
    df = load_data()
else:
    st.warning("⚠️ Please upload **energy_clean.csv** using the sidebar uploader.")
    st.info("💡 Run **P9_clean_data.py** in Jupyter first → saves energy_clean.csv → upload here.")
    st.stop()
    df = pd.DataFrame()

if df is None or df.empty:
    st.warning("⚠️ Dataset is empty — check energy_clean.csv.")
    st.stop()

S["df_work"] = df

TARGET  = "high_consumption"
REG_T   = "Global_active_power"

NUM_COLS = [c for c in ["Global_active_power","Global_reactive_power","Voltage",
                         "Global_intensity","Sub_metering_1","Sub_metering_2",
                         "Sub_metering_3","Total_submetering","Unmetered_power",
                         "Power_factor","Rolling_24h_mean","Rolling_7d_mean"]
            if c in df.columns]
TIME_COLS = [c for c in ["Hour","DayOfWeek","Month","Weekend","is_peak"] if c in df.columns]

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Data Overview",
    "2 · Power Distribution",
    "3 · Time Series ★",
    "4 · Hourly Patterns ★",
    "5 · Seasonal Analysis ★",
    "6 · Sub-metering ★",
    "7 · STL Decomposition ★",
    "8 · Anomaly Detection ★",
    "9 · Feature Engineering",
    "10 · Correlation",
    "11 · A/B Test ★",
    "12 · Multicollinearity",
    "13 · Insights & Report",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("📋 Tab 1 — Data Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Hourly Records", f"{len(df):,}")
    c2.metric("Features",       f"{df.shape[1]}")
    c3.metric("Date Range",     f"{df['DateTime'].dt.year.min()}–{df['DateTime'].dt.year.max()}" if "DateTime" in df.columns else "N/A")
    c4.metric("Avg Power",      f"{df[REG_T].mean():.3f} kW" if REG_T in df.columns else "N/A")
    c5.metric("Max Power",      f"{df[REG_T].max():.3f} kW" if REG_T in df.columns else "N/A")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📄 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        sec("📐 Column Info")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  df.dtypes.astype(str).values,
            "Nulls":  df.isnull().sum().values,
        })
        st.dataframe(info_df, use_container_width=True)

    st.markdown("---")
    sec("📊 Descriptive Statistics")
    st.dataframe(df[NUM_COLS].describe().round(4), use_container_width=True)

    st.markdown("---")
    sec("🗂 Data Dictionary")
    dd = pd.DataFrame({
        "Column": ["Global_active_power","Global_reactive_power","Voltage",
                   "Global_intensity","Sub_metering_1","Sub_metering_2",
                   "Sub_metering_3","Total_submetering","Unmetered_power",
                   "Power_factor","Hour","DayOfWeek","Month","Weekend",
                   "is_peak","Season","Period",
                   "Rolling_24h_mean","Rolling_7d_mean","high_consumption"],
        "Unit/Type": ["kW","kW","V","A","Wh","Wh","Wh","Wh","Wh",
                      "ratio","0–23","0–6","1–12","0/1","0/1",
                      "category","category","kW","kW","0/1"],
        "Description": [
            "Household global active power — regression target",
            "Household global reactive power",
            "Minute-averaged voltage",
            "Household global intensity (current)",
            "Sub-meter 1: kitchen (dishwasher, oven, microwave)",
            "Sub-meter 2: laundry (washing machine, dryer, fridge)",
            "Sub-meter 3: water heater + air conditioning",
            "Sum of all three sub-meters",
            "Power not captured by sub-meters (other appliances)",
            "Active / (Active + Reactive) — efficiency proxy",
            "Hour of day","Day of week (0=Mon)","Month","1=Weekend",
            "1=Peak hour (7–9AM or 6–9PM)",
            "Spring/Summer/Autumn/Winter",
            "Morning/Afternoon/Evening/Night",
            "24-hour rolling mean of Global_active_power",
            "7-day (168h) rolling mean of Global_active_power",
            "1 if Global_active_power > median — classification target",
        ]
    })
    st.dataframe(dd, use_container_width=True)
    info("Data resampled from minute to hourly frequency — reduces 1M rows to ~17K while preserving patterns.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — POWER DISTRIBUTION
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📊 Tab 2 — Power Distribution")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean Power",   f"{df[REG_T].mean():.4f} kW")
    c2.metric("Median Power", f"{df[REG_T].median():.4f} kW")
    c3.metric("Std Dev",      f"{df[REG_T].std():.4f} kW")
    c4.metric("Max Power",    f"{df[REG_T].max():.4f} kW")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Histogram — Global Active Power")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(df[REG_T].dropna(), bins=60, color=CLR["amber"],
                edgecolor="white", alpha=0.85)
        ax.axvline(df[REG_T].mean(),   color=CLR["danger"], lw=2, ls="--",
                   label=f"Mean={df[REG_T].mean():.3f}")
        ax.axvline(df[REG_T].median(), color=CLR["primary"], lw=2, ls="--",
                   label=f"Median={df[REG_T].median():.3f}")
        ax.set_xlabel("Global Active Power (kW)"); ax.set_ylabel("Count")
        ax.set_title("Distribution of Hourly Active Power"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📦 Box Plot — All Power Features")
        power_cols = [c for c in ["Global_active_power","Global_reactive_power",
                                   "Global_intensity"] if c in df.columns]
        fig2, ax2 = plt.subplots(figsize=(7,4))
        bp = ax2.boxplot([df[c].dropna() for c in power_cols],
                         patch_artist=True, labels=power_cols)
        colors_bp = [CLR["amber"], CLR["teal"], CLR["primary"]]
        for patch, color in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        for med in bp["medians"]: med.set_color(CLR["danger"]); med.set_linewidth(2)
        ax2.set_title("Power Feature Distributions"); ax2.set_xticklabels(power_cols, rotation=15)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 High vs Low Consumption Balance")
    if TARGET in df.columns:
        bal = df[TARGET].value_counts().reset_index()
        bal.columns = ["Class","Count"]
        bal["Label"] = bal["Class"].map({0:"Low Consumption",1:"High Consumption"})
        bal["Pct"]   = (bal["Count"]/len(df)*100).round(1)
        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.bar(bal, x="Label", y="Count", color="Label",
                          color_discrete_map={"Low Consumption":CLR["primary"],
                                              "High Consumption":CLR["danger"]},
                          text=bal["Pct"].apply(lambda x: f"{x}%"),
                          title="Class Balance — high_consumption target")
            fig3.update_traces(textposition="outside")
            fig3.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            fig4 = px.pie(bal, names="Label", values="Count", hole=0.45,
                          color="Label",
                          color_discrete_map={"Low Consumption":CLR["primary"],
                                              "High Consumption":CLR["danger"]},
                          title="~50/50 Balance")
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, use_container_width=True)

    insight("Right-skewed distribution — occasional very high power events pull the mean above median.")
    insight("~50/50 balance by design (median split) — no class_weight adjustment needed.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — TIME SERIES ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("📈 Tab 3 — Time Series ★")
    info("Full consumption timeline with 7-day rolling average — reveals trends and seasonality.")

    if "DateTime" in df.columns:
        sec("📈 Full Timeline — Global Active Power")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["DateTime"], y=df[REG_T],
            name="Hourly Power", mode="lines",
            line=dict(color=CLR["amber"], width=0.8), opacity=0.6))
        if "Rolling_7d_mean" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["DateTime"], y=df["Rolling_7d_mean"],
                name="7-Day Rolling Avg", mode="lines",
                line=dict(color=CLR["danger"], width=2.5)))
        if "Rolling_24h_mean" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["DateTime"], y=df["Rolling_24h_mean"],
                name="24H Rolling Avg", mode="lines",
                line=dict(color=CLR["primary"], width=1.5), opacity=0.8))
        fig.update_layout(
            title="Household Power Consumption — Full Timeline",
            xaxis_title="Date", yaxis_title="Global Active Power (kW)",
            height=450, hovermode="x unified", legend=dict(x=0.01,y=0.99))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        sec("📊 Daily Average Consumption")
        daily = df.set_index("DateTime")[REG_T].resample("D").mean().reset_index()
        daily.columns = ["Date","Avg_Power"]
        fig2 = px.line(daily, x="Date", y="Avg_Power",
                       title="Daily Average Active Power",
                       labels={"Avg_Power":"Avg Power (kW)"},
                       color_discrete_sequence=[CLR["teal"]])
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    insight("Clear seasonal pattern — winter months show higher consumption (heating demand).")
    insight("7-day rolling average smooths daily noise and reveals the underlying trend cleanly.")
    warn("Sharp spikes in the timeline are anomaly candidates — investigated in Tab 8.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — HOURLY PATTERNS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("⏰ Tab 4 — Hourly Patterns ★")
    info("Energy consumption follows strong intraday patterns — two distinct peaks.")

    if "Hour" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            sec("📊 Average Power by Hour")
            hourly = df.groupby("Hour")[REG_T].mean().reset_index()
            fig = px.bar(hourly, x="Hour", y=REG_T,
                         color=REG_T, color_continuous_scale=["#e3f2fd","#f57f17","#c62828"],
                         title="Average Active Power by Hour of Day",
                         labels={REG_T:"Avg Power (kW)"})
            peak_hours = [7,8,9,18,19,20,21]
            for ph in peak_hours:
                fig.add_vline(x=ph, line_dash="dot", line_color="red", opacity=0.4)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sec("📊 Peak vs Off-Peak Distribution")
            if "is_peak" in df.columns:
                peak_comp = df.groupby("is_peak")[REG_T].agg(
                    Mean="mean", Median="median", Std="std").reset_index()
                peak_comp["Label"] = peak_comp["is_peak"].map({0:"Off-Peak",1:"Peak Hours"})
                fig2 = px.bar(peak_comp, x="Label", y="Mean",
                              color="Label",
                              color_discrete_map={"Off-Peak":CLR["primary"],
                                                  "Peak Hours":CLR["danger"]},
                              error_y="Std",
                              title="Avg Power: Peak vs Off-Peak",
                              text=peak_comp["Mean"].apply(lambda x: f"{x:.3f} kW"))
                fig2.update_traces(textposition="outside")
                fig2.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        sec("🔥 Heatmap — Hour × Day of Week ★")
        if "DayOfWeek" in df.columns:
            heat = df.groupby(["DayOfWeek","Hour"])[REG_T].mean().unstack()
            day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            fig3, ax = plt.subplots(figsize=(14,5))
            sns.heatmap(heat, cmap="YlOrRd", ax=ax, linewidths=0,
                        xticklabels=list(range(24)),
                        yticklabels=day_labels)
            ax.set_title("Average Power (kW) — Hour × Day of Week", fontsize=13)
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Day of Week")
            plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Two clear consumption peaks: morning (7–9 AM) and evening (6–9 PM) — matches human activity patterns.")
    insight("Weekend patterns differ from weekdays — later morning peak, more uniform afternoon.")
    warn("Peak hours consume significantly more power — demand-response programs should target 6–9 PM.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — SEASONAL ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("🌡 Tab 5 — Seasonal Analysis ★")
    info("Winter heating dominates energy consumption — strong seasonal signal.")

    if "Month" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            sec("📊 Average Power by Month")
            monthly = df.groupby("Month")[REG_T].mean().reset_index()
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            monthly["Month_Name"] = monthly["Month"].map(month_names)
            fig = px.bar(monthly, x="Month_Name", y=REG_T,
                         color=REG_T, color_continuous_scale=["#e3f2fd","#f57f17","#c62828"],
                         title="Average Active Power by Month",
                         labels={REG_T:"Avg Power (kW)"})
            fig.update_layout(height=380,
                              xaxis=dict(categoryorder="array",
                                        categoryarray=list(month_names.values())))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sec("📊 Average Power by Season")
            if "Season" in df.columns:
                seasonal = df.groupby("Season")[REG_T].agg(
                    Mean="mean", Std="std").reset_index()
                season_order = ["Winter","Spring","Summer","Autumn"]
                seasonal["Season"] = pd.Categorical(seasonal["Season"],
                                                     categories=season_order, ordered=True)
                seasonal = seasonal.sort_values("Season")
                fig2 = px.bar(seasonal, x="Season", y="Mean",
                              color="Season",
                              color_discrete_map={"Winter":CLR["primary"],
                                                  "Spring":CLR["success"],
                                                  "Summer":CLR["amber"],
                                                  "Autumn":CLR["warning"]},
                              error_y="Std",
                              title="Avg Power by Season",
                              text=seasonal["Mean"].apply(lambda x: f"{x:.3f} kW"))
                fig2.update_traces(textposition="outside")
                fig2.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        sec("📊 Period of Day Analysis")
        if "Period" in df.columns:
            period_df = df.groupby("Period")[REG_T].mean().reset_index()
            period_order = ["Morning","Afternoon","Evening","Night"]
            period_df["Period"] = pd.Categorical(period_df["Period"],
                                                   categories=period_order, ordered=True)
            period_df = period_df.sort_values("Period")
            col3, col4 = st.columns(2)
            with col3:
                st.dataframe(period_df.round(4), use_container_width=True)
            with col4:
                fig3 = px.bar(period_df, x="Period", y=REG_T,
                              color="Period",
                              color_discrete_map={"Morning":CLR["amber"],
                                                  "Afternoon":CLR["teal"],
                                                  "Evening":CLR["danger"],
                                                  "Night":CLR["dark"]},
                              title="Avg Power by Period of Day",
                              text=period_df[REG_T].apply(lambda x: f"{x:.3f}"))
                fig3.update_traces(textposition="outside")
                fig3.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

    insight("Winter consumption is highest — heating is the dominant energy driver in this household.")
    insight("Evening is the highest-consumption period of the day — consistent across all seasons.")
    warn("Summer is relatively low — this household likely uses gas for heating, not electric.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — SUB-METERING ★
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("🔌 Tab 6 — Sub-metering Analysis ★")
    info("Sub-meters measure specific appliance zones — reveals WHICH devices drive consumption.")

    sub_cols = [c for c in ["Sub_metering_1","Sub_metering_2","Sub_metering_3"] if c in df.columns]
    labels   = {"Sub_metering_1":"Kitchen","Sub_metering_2":"Laundry","Sub_metering_3":"HVAC/Water"}

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Kitchen (avg Wh/h)",  f"{df['Sub_metering_1'].mean():.2f}" if "Sub_metering_1" in df.columns else "N/A")
    c2.metric("Laundry (avg Wh/h)",  f"{df['Sub_metering_2'].mean():.2f}" if "Sub_metering_2" in df.columns else "N/A")
    c3.metric("HVAC (avg Wh/h)",     f"{df['Sub_metering_3'].mean():.2f}" if "Sub_metering_3" in df.columns else "N/A")
    c4.metric("Unmetered (avg Wh/h)",f"{df['Unmetered_power'].mean():.2f}" if "Unmetered_power" in df.columns else "N/A")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Sub-meter Totals (kWh)")
        sub_totals = pd.DataFrame({
            "Zone":  ["Kitchen","Laundry","HVAC/Water","Unmetered"],
            "Total_Wh": [df["Sub_metering_1"].sum() if "Sub_metering_1" in df.columns else 0,
                         df["Sub_metering_2"].sum() if "Sub_metering_2" in df.columns else 0,
                         df["Sub_metering_3"].sum() if "Sub_metering_3" in df.columns else 0,
                         df["Unmetered_power"].sum() if "Unmetered_power" in df.columns else 0]
        })
        sub_totals["Total_kWh"] = (sub_totals["Total_Wh"]/1000).round(1)
        sub_totals["Share%"]    = (sub_totals["Total_Wh"]/sub_totals["Total_Wh"].sum()*100).round(1)
        fig = px.pie(sub_totals, names="Zone", values="Total_kWh",
                     color_discrete_sequence=[CLR["amber"],CLR["teal"],CLR["primary"],CLR["grey"]],
                     title="Energy Share by Zone", hole=0.4)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Sub-meter Hourly Profile")
        hourly_sub = df.groupby("Hour")[sub_cols].mean().reset_index()
        fig2 = go.Figure()
        colors_sub = [CLR["amber"],CLR["teal"],CLR["primary"]]
        for col_name, color in zip(sub_cols, colors_sub):
            fig2.add_trace(go.Scatter(
                x=hourly_sub["Hour"], y=hourly_sub[col_name],
                name=labels.get(col_name, col_name),
                mode="lines+markers", line=dict(color=color, width=2)))
        fig2.update_layout(
            title="Sub-meter Avg Wh/h by Hour",
            xaxis_title="Hour", yaxis_title="Wh/hour",
            height=380, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Sub-meter Seasonal Profile")
    if "Season" in df.columns:
        seasonal_sub = df.groupby("Season")[sub_cols].mean().reset_index()
        fig3 = go.Figure()
        for col_name, color in zip(sub_cols, colors_sub):
            fig3.add_trace(go.Bar(name=labels.get(col_name,col_name),
                                  x=seasonal_sub["Season"],
                                  y=seasonal_sub[col_name],
                                  marker_color=color))
        fig3.update_layout(barmode="group", height=380,
                           title="Sub-meter Avg Wh/h by Season",
                           yaxis_title="Wh/hour")
        st.plotly_chart(fig3, use_container_width=True)

    if "Unmetered_power" in df.columns:
        unmet_pct = df["Unmetered_power"].sum() / (df["Total_submetering"].sum() + df["Unmetered_power"].sum()) * 100
        insight(f"Unmetered power accounts for ~{unmet_pct:.1f}% of total — significant 'other appliances' usage.")
    insight("HVAC/Water heater (Sub_metering_3) shows strong winter peak — confirms heating is electric.")
    warn("Kitchen sub-meter peaks at breakfast and dinner times — consistent with cooking activity.")

# ══════════════════════════════════════════════════════════════
# TAB 7 — STL DECOMPOSITION ★
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("📉 Tab 7 — STL Time Series Decomposition ★")
    info("STL splits the time series into Trend + Seasonality + Residual — reveals hidden structure.")

    if "DateTime" in df.columns:
        daily = df.set_index("DateTime")[REG_T].resample("D").mean().dropna()

        try:
            decomp = seasonal_decompose(daily, model="additive", period=365)

            fig, axes = plt.subplots(4, 1, figsize=(14,12), sharex=True)
            axes[0].plot(daily.index, daily.values, color=CLR["amber"], lw=1, label="Original")
            axes[0].set_title("Original — Daily Global Active Power", fontsize=11)
            axes[0].set_ylabel("kW"); axes[0].legend()

            axes[1].plot(decomp.trend.index, decomp.trend.values, color=CLR["danger"], lw=2, label="Trend")
            axes[1].set_title("Trend Component", fontsize=11)
            axes[1].set_ylabel("kW"); axes[1].legend()

            axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color=CLR["teal"], lw=1, label="Seasonality")
            axes[2].set_title("Seasonal Component (annual)", fontsize=11)
            axes[2].set_ylabel("kW"); axes[2].legend()

            axes[3].plot(decomp.resid.index, decomp.resid.values, color=CLR["grey"],
                         lw=0.8, alpha=0.7, label="Residual")
            axes[3].axhline(0, color="black", lw=1, ls="--")
            axes[3].set_title("Residual Component", fontsize=11)
            axes[3].set_ylabel("kW"); axes[3].legend()

            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        except Exception as e:
            warn(f"STL decomposition error: {e}. Try uploading a longer time series.")

        st.markdown("---")
        sec("📊 Weekly Seasonality (7-day pattern)")
        weekly = df.groupby("DayOfWeek")[REG_T].mean().reset_index()
        day_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        weekly["Day"] = weekly["DayOfWeek"].map(day_names)
        fig2 = px.line(weekly, x="Day", y=REG_T, markers=True,
                       title="Average Power by Day of Week (Weekly Seasonality)",
                       labels={REG_T:"Avg Power (kW)"},
                       color_discrete_sequence=[CLR["danger"]])
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    insight("Trend component reveals long-term drift — useful for capacity planning.")
    insight("Seasonal component confirms annual cycle — winter peaks, summer troughs.")
    warn("Large residuals indicate anomalous days — equipment faults, unusual events, or data errors.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — ANOMALY DETECTION ★
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("🚨 Tab 8 — Anomaly Detection ★")
    info("Z-score and IQR methods identify abnormal consumption hours — spikes or outages.")

    z = np.abs(zscore(df[REG_T].dropna()))
    outlier_mask = z > 3
    n_out = outlier_mask.sum()

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Outliers (|Z|>3)", f"{n_out:,}")
    c2.metric("Outlier Rate",            f"{n_out/len(df)*100:.2f}%")
    c3.metric("Max Z-Score",             f"{z.max():.2f}")

    sec("📈 Z-Score Timeline — Anomaly Points Highlighted")
    if "DateTime" in df.columns:
        dt_idx = df["DateTime"].iloc[df[REG_T].dropna().index]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dt_idx, y=df[REG_T].dropna().values,
            mode="lines", name="Power", line=dict(color=CLR["amber"], width=1), opacity=0.6))
        outlier_vals = df[REG_T].dropna()[outlier_mask]
        outlier_dt   = dt_idx[outlier_mask]
        fig.add_trace(go.Scatter(
            x=outlier_dt, y=outlier_vals,
            mode="markers", name="Anomaly",
            marker=dict(color=CLR["danger"], size=7, symbol="x")))
        fig.update_layout(title="Power Timeline with Anomalies (|Z|>3)",
                          xaxis_title="DateTime", yaxis_title="kW",
                          height=420, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    sec("📊 IQR Outliers — All Numeric Features")
    iqr_rows = []
    for col in [REG_T,"Sub_metering_1","Sub_metering_2","Sub_metering_3"]:
        if col not in df.columns: continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        out = df[(df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)]
        iqr_rows.append({"Feature":col, "Outliers":len(out),
                         "Outlier%":round(len(out)/len(df)*100,2),
                         "Lower":round(Q1-1.5*IQR,3), "Upper":round(Q3+1.5*IQR,3)})
    st.dataframe(pd.DataFrame(iqr_rows), use_container_width=True)

    st.markdown("---")
    sec("📊 Anomaly Hours Distribution")
    if "Hour" in df.columns and n_out > 0:
        anom_hours = df.iloc[df[REG_T].dropna().index[outlier_mask]]["Hour"].value_counts().sort_index().reset_index()
        anom_hours.columns = ["Hour","Count"]
        fig2 = px.bar(anom_hours, x="Hour", y="Count",
                      color="Count", color_continuous_scale=["#fff3e0","#c62828"],
                      title="Anomaly Count by Hour of Day")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    insight(f"{n_out} anomalous hours detected — investigate for equipment faults or unusual events.")
    warn("Anomalies during night hours (0–5 AM) suggest potential equipment malfunction — no normal usage expected.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("⚙️ Tab 9 — Feature Engineering")

    fe = pd.DataFrame({
        "Feature":     ["Hour","DayOfWeek","Month","Weekend","is_peak",
                        "Season","Period","Total_submetering",
                        "Unmetered_power","Power_factor",
                        "Rolling_24h_mean","Rolling_7d_mean"],
        "Source":      ["DateTime.hour","DateTime.dayofweek","DateTime.month",
                        "DayOfWeek≥5","Hour∈{7,8,9,18,19,20,21}",
                        "Month binned","Hour binned",
                        "Sum S1+S2+S3","(kW×1000/60)−Total_sub",
                        "Active/(Active+Reactive)",
                        "24-hour rolling mean","168-hour rolling mean"],
        "Reason":      ["Intraday consumption pattern",
                        "Weekday vs weekend behavior",
                        "Monthly seasonality",
                        "Weekend consumption differs",
                        "Peak demand hours flag",
                        "Annual seasonality capture",
                        "Day part consumption pattern",
                        "Known appliance energy usage",
                        "Unknown appliance estimation",
                        "Electrical efficiency proxy",
                        "Short-term trend feature",
                        "Long-term trend feature"],
    })
    st.dataframe(fe, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Rolling Averages")
        if "DateTime" in df.columns and "Rolling_24h_mean" in df.columns:
            sample = df.iloc[::6].copy()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sample["DateTime"], y=sample[REG_T],
                                     name="Hourly", opacity=0.4,
                                     line=dict(color=CLR["grey"], width=0.8)))
            fig.add_trace(go.Scatter(x=sample["DateTime"], y=sample["Rolling_24h_mean"],
                                     name="24H Rolling", line=dict(color=CLR["primary"], width=1.5)))
            if "Rolling_7d_mean" in df.columns:
                fig.add_trace(go.Scatter(x=sample["DateTime"], y=sample["Rolling_7d_mean"],
                                         name="7D Rolling", line=dict(color=CLR["danger"], width=2.5)))
            fig.update_layout(title="Rolling Averages vs Raw",
                               height=380, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Unmetered Power Distribution")
        if "Unmetered_power" in df.columns:
            fig2, ax = plt.subplots(figsize=(7,4))
            ax.hist(df["Unmetered_power"].clip(0,3), bins=50,
                    color=CLR["purple"], alpha=0.8, edgecolor="white")
            ax.set_xlabel("Unmetered Power (Wh/h)"); ax.set_ylabel("Count")
            ax.set_title("Distribution of Unmetered Power")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("Rolling averages are strong ML features — they encode temporal context without leaking future data.")
    insight("Unmetered power often has the largest share — other appliances (TV, computers, lights) dominate.")

# ══════════════════════════════════════════════════════════════
# TAB 10 — CORRELATION
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    sec("🔥 Tab 10 — Correlation Analysis")

    corr_cols = [c for c in NUM_COLS + TIME_COLS + [TARGET] if c in df.columns]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13,9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.4, annot_kws={"size":7})
    ax.set_title("Correlation Matrix — Energy Consumption Features", fontsize=13, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sec("🎯 Top Correlations with Global_active_power")
    if REG_T in corr.columns:
        tgt = corr[REG_T].drop(REG_T).sort_values(key=abs, ascending=False).head(12)
        fig2, ax2 = plt.subplots(figsize=(10,5))
        colors_bar = [CLR["danger"] if v>0 else CLR["success"] for v in tgt.values]
        ax2.barh(tgt.index, tgt.values, color=colors_bar)
        ax2.axvline(0, color="black", lw=0.8)
        ax2.set_xlabel("Pearson r with Global_active_power")
        ax2.set_title("Feature Correlation with Power Target")
        for i,(idx,val) in enumerate(tgt.items()):
            ax2.text(val+0.005 if val>=0 else val-0.005, i,
                     f"{val:.3f}", va="center",
                     ha="left" if val>=0 else "right", fontsize=9)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("Global_intensity has near-perfect correlation with Global_active_power (P=V×I) — expected physically.")
    insight("Rolling averages show strong correlation — temporal context is highly predictive.")
    warn("Global_intensity and Global_active_power are nearly redundant — consider dropping one for linear models.")

# ══════════════════════════════════════════════════════════════
# TAB 11 — A/B TEST ★
# ══════════════════════════════════════════════════════════════
with tabs[10]:
    sec("🧪 Tab 11 — A/B Test: Peak vs Off-Peak ★")
    info("Hypothesis: Peak hours (7–9 AM, 6–9 PM) consume significantly more power than off-peak hours.")

    if "is_peak" in df.columns:
        gA = df[df["is_peak"]==0][REG_T].dropna()
        gB = df[df["is_peak"]==1][REG_T].dropna()

        t_stat, p_val = stats.ttest_ind(gA, gB, equal_var=False)
        pooled        = np.sqrt((gA.std()**2 + gB.std()**2) / 2)
        cohens_d      = (gB.mean() - gA.mean()) / pooled
        diff          = gB.mean() - gA.mean()
        se            = np.sqrt(gA.var()/len(gA) + gB.var()/len(gB))
        ci_lo, ci_hi  = diff - 1.96*se, diff + 1.96*se

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Off-Peak Avg (kW)",  f"{gA.mean():.4f}")
        c2.metric("Peak Avg (kW)",      f"{gB.mean():.4f}")
        c3.metric("p-value",            f"{p_val:.6f}")
        c4.metric("Cohen's d",          f"{cohens_d:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(gA, bins=50, alpha=0.6, color=CLR["primary"],
                    label=f"Off-Peak (n={len(gA):,})", density=True)
            ax.hist(gB, bins=50, alpha=0.6, color=CLR["danger"],
                    label=f"Peak (n={len(gB):,})", density=True)
            ax.axvline(gA.mean(), color=CLR["primary"], lw=2, ls="--")
            ax.axvline(gB.mean(), color=CLR["danger"],  lw=2, ls="--")
            ax.set_xlabel("Global Active Power (kW)"); ax.legend()
            ax.set_title("Distribution: Peak vs Off-Peak")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            bp = ax2.boxplot([gA, gB], patch_artist=True,
                             tick_labels=["Off-Peak","Peak"])
            bp["boxes"][0].set_facecolor(CLR["light"])
            bp["boxes"][1].set_facecolor("#fce4ec")
            for m in bp["medians"]: m.set_color(CLR["danger"]); m.set_linewidth(2)
            ax2.set_ylabel("Power (kW)"); ax2.set_title("Box Plot: Peak vs Off-Peak")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("---")
        sec("📋 Test Results")
        res = pd.DataFrame({
            "Metric": ["Test","H₀","H₁","t-statistic","p-value",
                       "Significant (α=0.05)","Cohen's d","Effect Size",
                       "95% CI","Decision"],
            "Result": [
                "Welch T-Test (unequal variance)",
                "Peak power = Off-peak power",
                "Peak power ≠ Off-peak power",
                f"{t_stat:.4f}", f"{p_val:.6f}",
                "✅ YES" if p_val < 0.05 else "❌ NO",
                f"{cohens_d:.4f}",
                "Large" if abs(cohens_d)>0.8 else "Medium" if abs(cohens_d)>0.5 else "Small",
                f"[{ci_lo:.4f}, {ci_hi:.4f}]",
                "✅ REJECT H₀" if p_val < 0.05 else "❌ FAIL to reject H₀"
            ]
        })
        st.dataframe(res, use_container_width=True)

        if p_val < 0.05:
            pct_diff = (gB.mean()-gA.mean())/gA.mean()*100
            insight(f"Peak hours use {pct_diff:+.1f}% more power than off-peak — statistically significant.")
            warn("Demand response programs shifting load from peak to off-peak would reduce grid stress significantly.")

# ══════════════════════════════════════════════════════════════
# TAB 12 — MULTICOLLINEARITY
# ══════════════════════════════════════════════════════════════
with tabs[11]:
    sec("🔁 Tab 12 — Multicollinearity / VIF")
    info("VIF > 10 = severe multicollinearity. Expected between power metrics (physical relationships).")

    vif_cols = [c for c in ["Global_active_power","Global_reactive_power",
                             "Global_intensity","Total_submetering",
                             "Unmetered_power","Hour","Month","Weekend","is_peak"]
                if c in df.columns]
    vif_data = df[vif_cols].dropna()
    try:
        vif_df = pd.DataFrame({
            "Feature": vif_cols,
            "VIF": [round(variance_inflation_factor(vif_data.values,i),2)
                    for i in range(len(vif_cols))]
        }).sort_values("VIF", ascending=False)
        vif_df["Risk"] = vif_df["VIF"].apply(
            lambda v: "🔴 High" if v>10 else "🟡 Medium" if v>5 else "🟢 Low")

        col1, col2 = st.columns([1,1.5])
        with col1:
            st.dataframe(vif_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7,5))
            colors_vif = [CLR["danger"] if v>10 else CLR["warning"] if v>5
                          else CLR["success"] for v in vif_df["VIF"]]
            ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors_vif)
            ax.axvline(10, color=CLR["danger"],  lw=2, ls="--", label="VIF=10")
            ax.axvline(5,  color=CLR["warning"], lw=1.5, ls=":",  label="VIF=5")
            ax.set_xlabel("VIF"); ax.set_title("Multicollinearity Check")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()
    except Exception as e:
        warn(f"VIF error: {e}")

    warn("Global_intensity is mathematically derived from power — expect high VIF. Drop for linear models.")
    insight("Tree-based models (RF, GB) are immune to multicollinearity — VIF only matters for linear regression.")

# ══════════════════════════════════════════════════════════════
# TAB 13 — INSIGHTS & REPORT
# ══════════════════════════════════════════════════════════════
with tabs[12]:
    sec("💡 Tab 13 — Insights & Recommendations")

    st.markdown(f"### ⚡ Energy Consumption — Final Report")
    st.markdown(f"**{len(df):,} hourly records · UCI Household Power · Dec 2006–Dec 2008 · M3**")
    st.markdown("---")

    sec("1️⃣ Consumption Patterns")
    insight("Two daily peaks: Morning (7–9 AM) and Evening (6–9 PM) — matches human activity cycle.")
    insight("Winter consumes significantly more than Summer — heating is the dominant driver.")
    insight("Weekday vs Weekend differ — later morning peak, more uniform weekend profile.")

    sec("2️⃣ Sub-metering Insights")
    insight("HVAC/Water heater (Sub_metering_3) spikes in winter — heating is electrically driven.")
    insight("Unmetered power is the largest share — lights, TV, computers dominate untracked usage.")
    warn("Kitchen and laundry peak at expected meal/washing times — predictable scheduling opportunity.")

    sec("3️⃣ Anomalies & Quality")
    insight("Anomalous hours detected via Z-score — mostly extreme winter evenings or equipment events.")
    warn("Night-time anomalies (12 AM–5 AM) suggest equipment running when it shouldn't — efficiency loss.")

    sec("4️⃣ Recommendations")
    recs = [
        ("⏰ Shift Load Off-Peak",    "Move dishwasher, washing machine, dryer to off-peak hours (10 PM–6 AM)."),
        ("❄️ Heating Optimization",   "Pre-heat home before peak hours, reduce thermostat during evening peak."),
        ("🔌 Smart Metering",         "Install sub-meters on unmetered circuits — 40%+ of usage is untracked."),
        ("🚨 Anomaly Alerts",         "Set automated alerts for consumption > 3σ above rolling mean."),
        ("📊 Demand Forecasting",     "Use ML regression model to forecast next-hour consumption for grid planning."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""ENERGY CONSUMPTION — FINAL REPORT
M3 · UCI Household Power · {len(df):,} Hourly Records
Period: Dec 2006 – Dec 2008

KEY PATTERNS:
1. Two daily peaks: 7-9 AM and 6-9 PM
2. Winter highest consumption (heating-driven)
3. Evening is highest-consumption period of day
4. Unmetered power = largest share (~40%+)
5. HVAC dominates winter sub-metering

ANOMALIES:
- Z-score outliers detected in timeline
- Night-time anomalies suggest equipment issues

RECOMMENDATIONS:
- Shift washer/dryer/dishwasher to 10PM-6AM
- Pre-heat before peak hours
- Install sub-meters on unmetered circuits
- Automated anomaly alerts (>3σ threshold)
- Deploy ML model for next-hour forecasting
"""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="EnergyConsumption_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        st.download_button("📥 Download Clean Data (.csv)",
                           df.to_csv(index=False),
                           file_name="energy_clean_M3.csv",
                           mime="text/csv", use_container_width=True)
