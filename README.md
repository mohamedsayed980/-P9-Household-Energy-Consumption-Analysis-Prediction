# ⚡ P9 — Household Energy Consumption Analysis & Prediction
**M3 · ML Engine Portfolio · Project 9 of 12**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Source-UCI_Repository-0052CC)](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

---

## 📌 Project Overview

End-to-end time series analysis and ML prediction on **~1 million minute-level electricity readings** from a single French household (Dec 2006 – Dec 2008), resampled to hourly frequency for ML modelling.

**Core Questions:**
- When does household power consumption peak during the day and year?
- Which appliance zones (kitchen, laundry, HVAC) drive the most energy?
- Can ML models predict hourly consumption and classify high-consumption periods?
- What anomalies exist in the timeline and what do they mean?

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| Raw Records | 1,048,575 minute readings |
| After Resampling | ~17,477 hourly records |
| Raw Features | 9 |
| After Engineering | 21 |
| Period | December 2006 – December 2008 |
| Compression | 60× reduction (minute → hourly) |

### Resampling Strategy
- **Power measurements** (Global_active_power, etc.) → **mean** per hour
- **Sub-metering** (S1, S2, S3) → **sum** per hour (Wh consumed)

---

## ⚠️ Data Quality Issues Fixed

| Issue | Fix |
|-------|-----|
| `?` null markers in 4,069 rows | `na_values='?'` on load → median imputation |
| All numeric columns stored as object | `pd.to_numeric(errors='coerce')` |
| 1M rows too large for ML | Resampled to hourly (~17K rows) |

---

## 🎯 Targets

| Type | Column | Description |
|------|--------|-------------|
| **Regression** | `Global_active_power` | Hourly avg active power (kW) |
| **Classification** | `high_consumption` | 1 if power > median (0.767 kW) |

**Balance:** ~50/50 by median split → **No class_weight adjustment needed**

**ADF Stationarity:** Global_active_power is **STATIONARY** (p≈0.000) → direct ML on values is valid

---

## ⚙️ Feature Engineering

| Feature | Source | Purpose |
|---------|--------|---------|
| `Hour` | DateTime.hour | Intraday consumption pattern |
| `DayOfWeek` | DateTime.dayofweek | Weekday vs weekend behaviour |
| `Month` | DateTime.month | Monthly seasonality |
| `Weekend` | DayOfWeek ≥ 5 | Weekend consumption differs |
| `is_peak` | Hour ∈ {7,8,9,18,19,20,21} | Peak demand hours flag |
| `Season` | Month binned | Annual seasonality capture |
| `Period` | Hour binned | Morning/Afternoon/Evening/Night |
| `Total_submetering` | S1+S2+S3 | Known appliance energy total |
| `Unmetered_power` | (kW×1000/60)−Total_sub | Unknown appliance estimation |
| `Power_factor` | Active/(Active+Reactive) | Electrical efficiency proxy |
| `Rolling_24h_mean` | 24-hour rolling mean | Short-term trend feature |
| `Rolling_7d_mean` | 168-hour rolling mean | Long-term trend feature |
| `high_consumption` | Power > median | Classification target |

---

## 📊 EDA Dashboard — 13 Tabs

| Tab | Title | Highlight |
|-----|-------|-----------|
| 1 | Data Overview | Shape, types, stats, dictionary |
| 2 | Power Distribution | Histogram, box plot, class balance |
| 3 | Time Series ★ | Full timeline + 7-day rolling average |
| 4 | Hourly Patterns ★ | Hour × Day heatmap — two daily peaks |
| 5 | Seasonal Analysis ★ | Monthly + seasonal profiles |
| 6 | Sub-metering ★ | Kitchen vs Laundry vs HVAC breakdown |
| 7 | STL Decomposition ★ | Trend + Seasonal + Residual |
| 8 | Anomaly Detection ★ | Z-score spikes on timeline |
| 9 | Feature Engineering | Rolling features + unmetered power |
| 10 | Correlation | Heatmap + top power predictors |
| 11 | A/B Test ★ | Peak vs Off-peak — Welch T-test + Cohen's d |
| 12 | Multicollinearity | VIF analysis |
| 13 | Insights & Report | Findings + recommendations + download |

---

## 🤖 ML Models — 5 Tabs

| Tab | Content |
|-----|---------|
| 1 | Training — 6 Reg + 6 Clf · individual buttons |
| 2 | Regression Results — R², MAE, RMSE |
| 3 | Classification Results — F1, Accuracy, ROC-AUC |
| 4 | Feature Importance — top energy predictors |
| 5 | Interactive Predict — next-hour consumption forecast |

**Regression (6):** Linear · Ridge · Lasso · Decision Tree · Random Forest · Gradient Boosting

**Classification (6):** Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · SVM (Linear) · KNN

---

## 🔑 Key Findings

**1. Two Daily Peaks**
Morning (7–9 AM) and Evening (6–9 PM) — matches human activity cycles. A/B test confirms peak hours consume significantly more power (p≈0.000).

**2. Winter Dominates**
Winter consumption is highest — HVAC/Water heater (Sub_metering_3) spikes in cold months, confirming electric heating.

**3. Unmetered Power is Largest Share**
~40%+ of consumption comes from appliances NOT measured by sub-meters — lights, TV, computers, other devices.

**4. Stationary Series**
Unlike stock prices, household power is stationary — direct ML on power values is valid without differencing.

**5. STL Reveals Clear Annual Cycle**
Trend is relatively flat (stable household), but seasonal component shows strong Winter > Summer pattern.

---

## 💡 Recommendations

| Priority | Action |
|----------|--------|
| ⏰ High | Shift dishwasher/washing machine to off-peak hours (10PM–6AM) |
| ❄️ High | Pre-heat before peak hours, reduce thermostat at 6PM |
| 🔌 Medium | Install sub-meters on unmetered circuits — 40%+ usage untracked |
| 🚨 Medium | Set automated alerts for consumption > 3σ above rolling mean |
| 📊 Low | Use ML regression model for next-hour demand forecasting |

---

## 🗂 Project Structure

```
📁 Repo_9_Energy_Consumption/
├── Home.py
├── M3_logo.png
├── requirements.txt
├── README.md
├── data/
│   └── energy_clean.csv          ← from P9_clean_data.py (Jupyter)
└── pages/
    ├── EDA_dashboard.py           ← 13-tab analysis
    └── ML_Models.py               ← 5-tab ML engine
```

---

## 🚀 How to Run

```bash
git clone https://github.com/YourUsername/Repo_9_Energy_Consumption.git
cd Repo_9_Energy_Consumption

pip install -r requirements.txt

# Step 1: Generate clean dataset in Jupyter
# Run P9_clean_data.py → saves energy_clean.csv
# Copy energy_clean.csv to data/ folder (do NOT open in Excel)

# Step 2: Launch app
streamlit run Home.py
```

> ⚠️ **Important:** Copy CSV directly via File Explorer — never open in Excel before copying. Excel corrupts CSV separators and encoding.

---

## 🛠 Tech Stack

`Python 3.11` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Scikit-learn` · `SciPy` · `Statsmodels` · `Psutil`

---

**Mohamed · M3 · ML Engine Portfolio — 12 End-to-End Data Science Projects**
