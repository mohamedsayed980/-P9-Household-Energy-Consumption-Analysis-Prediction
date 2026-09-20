"""
Repo_9_Energy_Consumption — ML_Models.py  (5 Tabs)
Author : Mohamed · M3
Regression     → Global_active_power
Classification → high_consumption  (~50/50 — no class_weight needed)
"""
import streamlit as st

import os, pathlib, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import psutil

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model      import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree              import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble          import (RandomForestRegressor, GradientBoostingRegressor,
                                       RandomForestClassifier, GradientBoostingClassifier)
from sklearn.svm               import LinearSVC
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.neighbors         import KNeighborsClassifier

warnings.filterwarnings("ignore")
S = st.session_state

st.set_page_config(page_title="ML Models · Energy · M3", page_icon="🤖", layout="wide")

LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"
DATA = pathlib.Path(__file__).parent.parent / "data" / "energy_clean.csv"

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🤖 ML Models")
    st.markdown("Energy Consumption · 5 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    _uploaded = st.file_uploader("Upload Clean CSV", type=["csv"], key="p9_ml_upload")
    if _uploaded is not None:
        st.success(f"✅ Using: {_uploaded.name}")
    else:
        st.info("Using default: energy_clean.csv")
    st.divider()
    st.markdown("### ⚙️ Options")
    test_size    = st.slider("Test Split %", 10, 40, 20, 5) / 100
    use_parallel = st.checkbox("Parallel (n_jobs=-1)", value=True)
    n_jobs       = -1 if use_parallel else 1
    st.info("Target ~50/50 balance\nNo class_weight needed")

CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","light":"#e3f2fd","dark":"#1a237e",
       "amber":"#f57f17","grey":"#546e7a","purple":"#6a1b9a"}

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
.insight-box p{color:#1b3a1f !important;margin:0;font-size:0.93rem;}
.warn-box{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.warn-box p{color:#4a2000 !important;margin:0;font-size:0.93rem;}
.info-box{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.info-box p{color:#0d2a4a !important;margin:0;font-size:0.93rem;}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)

def get_cpu_info(use_parallel, n_jobs):
    return {"total": os.cpu_count(), "used": n_jobs if use_parallel else 1,
            "percent": psutil.cpu_percent(interval=0.3)}

# ── LOAD ─────────────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes=None) -> pd.DataFrame:
    import io as _io
    if file_bytes is not None:
        df = pd.read_csv(_io.BytesIO(file_bytes), sep=",", decimal=".")
    else:
        df = pd.read_csv(DATA, sep=",", decimal=".")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    return df

_up = S.get("p9_ml_upload", None)
if _up is not None:
    _bytes = _up.read(); _up.seek(0)
    df = load_data(file_bytes=_bytes)
else:
    df = load_data()

if df.empty:
    st.warning("⚠️ No data. Run P9_clean_data.py then upload energy_clean.csv.")
    st.stop()

S["df_work"] = df

# ── FEATURE PREP ─────────────────────────────────────────────
REG_TARGET = "Global_active_power"
CLF_TARGET = "high_consumption"

FEAT_CANDIDATES = ["Global_reactive_power","Voltage","Global_intensity",
                   "Sub_metering_1","Sub_metering_2","Sub_metering_3",
                   "Total_submetering","Unmetered_power","Power_factor",
                   "Hour","DayOfWeek","Month","Weekend","is_peak",
                   "Rolling_24h_mean","Rolling_7d_mean"]

ALL_FEATS = [f for f in FEAT_CANDIDATES
             if f in df.columns and f not in [REG_TARGET, CLF_TARGET]]

df_ml = df[ALL_FEATS + [REG_TARGET, CLF_TARGET]].dropna().copy()
X     = df_ml[ALL_FEATS]
y_reg = df_ml[REG_TARGET]
y_clf = df_ml[CLF_TARGET]

X_train_r, X_test_r, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=test_size, random_state=42)
X_train_c, X_test_c, yc_train, yc_test = train_test_split(
    X, y_clf, test_size=test_size, random_state=42, stratify=y_clf)

scaler   = StandardScaler()
Xtr_r_sc = scaler.fit_transform(X_train_r)
Xte_r_sc = scaler.transform(X_test_r)
Xtr_c_sc = scaler.fit_transform(X_train_c)
Xte_c_sc = scaler.transform(X_test_c)

REG_MODELS = {
    "Linear Regression":  LinearRegression(),
    "Ridge":              Ridge(alpha=1.0),
    "Lasso":              Lasso(alpha=0.01, max_iter=5000),
    "Decision Tree":      DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
}
CLF_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=n_jobs, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (Linear)":        CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
    "KNN":                 KNeighborsClassifier(n_neighbors=7, n_jobs=n_jobs),
}

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs(["1 · Model Training",
                "2 · Regression Results",
                "3 · Classification Results",
                "4 · Feature Importance",
                "5 · Predict"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("🚀 Tab 1 — Model Training")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Records",      f"{len(df_ml):,}")
    c2.metric("Features",     f"{len(ALL_FEATS)}")
    c3.metric("Train Size",   f"{len(X_train_r):,}")
    c4.metric("Test Size",    f"{len(X_test_r):,}")
    c5.metric("Balance",      f"{y_clf.mean()*100:.1f}% high")

    cpu = get_cpu_info(use_parallel, n_jobs)
    st.info(f"🖥 CPU: {cpu['total']} cores · Using: {cpu['used']} · Load: {cpu['percent']}%")

    col1, col2 = st.columns(2)
    with col1:
        sec("🎯 Regression Target")
        st.markdown(f"**`{REG_TARGET}`** — predict hourly active power (kW)")
        st.markdown(f"Mean={y_reg.mean():.4f} · Range={y_reg.min():.4f}–{y_reg.max():.4f} kW")
    with col2:
        sec("🎯 Classification Target")
        st.markdown(f"**`{CLF_TARGET}`** — predict high consumption (0/1)")
        st.markdown(f"Balance: {y_clf.mean()*100:.1f}% high — no class_weight needed")

    if "reg_results" not in S: S["reg_results"] = []
    if "reg_models"  not in S: S["reg_models"]  = {}
    if "clf_results" not in S: S["clf_results"] = []
    if "clf_models"  not in S: S["clf_models"]  = {}
    S["X_test_r"] = X_test_r; S["Xte_r_sc"] = Xte_r_sc
    S["X_test_c"] = X_test_c; S["Xte_c_sc"] = Xte_c_sc
    S["yr_test"]  = yr_test;  S["yc_test"]  = yc_test
    S["scaler"]   = scaler;   S["X_cols"]   = ALL_FEATS

    def _done_r(n): return any(r["Model"]==n for r in S["reg_results"])
    def _done_c(n): return any(r["Model"]==n for r in S["clf_results"])

    def _train_reg(name, model):
        use_sc = name in ["Linear Regression","Ridge","Lasso"]
        Xtr = Xtr_r_sc if use_sc else X_train_r
        Xte = Xte_r_sc if use_sc else X_test_r
        t0  = time.time(); model.fit(Xtr, yr_train); preds = model.predict(Xte)
        row = {"Model":name,
               "R²":   round(r2_score(yr_test, preds),4),
               "MAE":  round(mean_absolute_error(yr_test, preds),4),
               "RMSE": round(np.sqrt(mean_squared_error(yr_test, preds)),4),
               "Time(s)": round(time.time()-t0,2)}
        S["reg_results"] = [r for r in S["reg_results"] if r["Model"]!=name] + [row]
        S["reg_models"][name] = model
        return row

    def _train_clf(name, model):
        use_sc = name in ["Logistic Regression","SVM (Linear)","KNN"]
        Xtr = Xtr_c_sc if use_sc else X_train_c
        Xte = Xte_c_sc if use_sc else X_test_c
        t0  = time.time(); model.fit(Xtr, yc_train); preds = model.predict(Xte)
        proba = model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
        row = {"Model":name,
               "Accuracy":  round(accuracy_score(yc_test, preds),4),
               "F1":        round(f1_score(yc_test, preds, zero_division=0),4),
               "Precision": round(precision_score(yc_test, preds, zero_division=0),4),
               "Recall":    round(recall_score(yc_test, preds, zero_division=0),4),
               "ROC-AUC":   round(roc_auc_score(yc_test,proba),4) if proba is not None else 0.0,
               "Time(s)":   round(time.time()-t0,2)}
        S["clf_results"] = [r for r in S["clf_results"] if r["Model"]!=name] + [row]
        S["clf_models"][name] = model
        return row

    st.markdown("---")
    sec("📈 Regression Models")
    info("Train each model one at a time.")
    rc = st.columns(3)
    for i,(name,model) in enumerate(REG_MODELS.items()):
        with rc[i%3]:
            label = f"✅ {name}" if _done_r(name) else f"▶ Train {name}"
            if st.button(label, key=f"reg_{name}", use_container_width=True):
                with st.spinner(f"Training {name}..."):
                    row = _train_reg(name, model)
                st.success(f"R²={row['R²']:.4f} · MAE={row['MAE']:.4f} · {row['Time(s)']}s")
                st.rerun()
            if _done_r(name):
                r = next(r for r in S["reg_results"] if r["Model"]==name)
                st.caption(f"R²={r['R²']:.4f} · MAE={r['MAE']:.4f} · {r['Time(s)']}s")

    if S["reg_results"]:
        st.dataframe(pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False)
                       .reset_index(drop=True)
                       .style.background_gradient(subset=["R²"],cmap="RdYlGn")
                       .format({"R²":"{:.4f}","MAE":"{:.4f}","RMSE":"{:.4f}"}),
                     use_container_width=True)

    st.markdown("---")
    sec("🎯 Classification Models")
    info("~50/50 balance — no class_weight needed. SVM (Linear) replaces RBF.")
    cc = st.columns(3)
    for i,(name,model) in enumerate(CLF_MODELS.items()):
        with cc[i%3]:
            label = f"✅ {name}" if _done_c(name) else f"▶ Train {name}"
            if st.button(label, key=f"clf_{name}", use_container_width=True):
                with st.spinner(f"Training {name}..."):
                    row = _train_clf(name, model)
                st.success(f"F1={row['F1']:.4f} · AUC={row['ROC-AUC']:.4f} · {row['Time(s)']}s")
                st.rerun()
            if _done_c(name):
                r = next(r for r in S["clf_results"] if r["Model"]==name)
                st.caption(f"F1={r['F1']:.4f} · AUC={r['ROC-AUC']:.4f} · {r['Time(s)']}s")

    if S["clf_results"]:
        st.dataframe(pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False)
                       .reset_index(drop=True)
                       .style.background_gradient(subset=["F1","ROC-AUC"],cmap="RdYlGn")
                       .format({c:"{:.4f}" for c in ["Accuracy","F1","Precision","Recall","ROC-AUC"]}),
                     use_container_width=True)

    n_done = len(S["reg_results"]) + len(S["clf_results"])
    st.info(f"📊 {n_done}/12 models trained." if n_done<12
            else "✅ All 12 models trained! Navigate to Results tabs →")

# ══════════════════════════════════════════════════════════════
# TAB 2 — REGRESSION RESULTS
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Regression Results")
    info("Predicting: **Global_active_power (kW)** · R² close to 1.0 expected with rolling features.")

    if not S.get("reg_results"):
        warn("Train at least one Regression model in Tab 1.")
    else:
        reg_df   = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True)
        best_reg = reg_df.iloc[0]["Model"]

        st.dataframe(reg_df.style.background_gradient(subset=["R²"],cmap="RdYlGn")
                                  .background_gradient(subset=["MAE","RMSE"],cmap="RdYlGn_r")
                                  .format({"R²":"{:.4f}","MAE":"{:.4f}","RMSE":"{:.4f}"}),
                     use_container_width=True)
        st.markdown(f"🏆 **Best:** `{best_reg}` — R²={reg_df.iloc[0]['R²']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(reg_df, x="Model", y="R²",
                         color="R²", color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="R² — All Regression Models",
                         text=reg_df["R²"].apply(lambda x: f"{x:.4f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="MAE",  x=reg_df["Model"], y=reg_df["MAE"],
                                  marker_color=CLR["warning"]))
            fig2.add_trace(go.Bar(name="RMSE", x=reg_df["Model"], y=reg_df["RMSE"],
                                  marker_color=CLR["danger"]))
            fig2.update_layout(barmode="group", height=370,
                                title="MAE vs RMSE", xaxis_tickangle=-25)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        sec(f"📈 Actual vs Predicted — {best_reg}")
        bm   = S["reg_models"][best_reg]
        Xte  = S["Xte_r_sc"] if best_reg in ["Linear Regression","Ridge","Lasso"] else S["X_test_r"]
        pred = bm.predict(Xte)
        col3, col4 = st.columns(2)
        with col3:
            fig3, ax = plt.subplots(figsize=(7,5))
            ax.scatter(yr_test, pred, alpha=0.3, s=8, color=CLR["amber"])
            lims = [min(yr_test.min(),pred.min()), max(yr_test.max(),pred.max())]
            ax.plot(lims, lims, "r--", lw=2, label="Perfect fit")
            ax.set_xlabel("Actual (kW)"); ax.set_ylabel("Predicted (kW)")
            ax.set_title(f"Actual vs Predicted — {best_reg}"); ax.legend()
            plt.tight_layout(); st.pyplot(fig3); plt.close()
        with col4:
            resid = yr_test.values - pred
            fig4, ax2 = plt.subplots(figsize=(7,5))
            ax2.scatter(pred, resid, alpha=0.3, s=8, color=CLR["teal"])
            ax2.axhline(0, color=CLR["danger"], lw=2, ls="--")
            ax2.set_xlabel("Predicted (kW)"); ax2.set_ylabel("Residual")
            ax2.set_title("Residual Plot")
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        insight(f"Best: {best_reg} · R²={reg_df.iloc[0]['R²']:.4f} — rolling features give strong R².")
        warn("High R² partly due to rolling mean features — they encode recent history. Check for leakage in production.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFICATION RESULTS
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("🎯 Tab 3 — Classification Results")
    info("Predicting: **high_consumption** · ~50/50 balance — accuracy is meaningful here.")

    if not S.get("clf_results"):
        warn("Train at least one Classification model in Tab 1.")
    else:
        clf_df   = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        best_clf = clf_df.iloc[0]["Model"]
        yc_test  = S["yc_test"]

        st.dataframe(clf_df.style.background_gradient(subset=["F1","ROC-AUC","Accuracy"],cmap="RdYlGn")
                                  .format({c:"{:.4f}" for c in ["Accuracy","F1","Precision","Recall","ROC-AUC"]}),
                     use_container_width=True)
        st.markdown(f"🏆 **Best:** `{best_clf}` — F1={clf_df.iloc[0]['F1']:.4f} · AUC={clf_df.iloc[0]['ROC-AUC']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(clf_df, x="Model", y="F1",
                         color="F1", color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="F1 Score — All Classifiers",
                         text=clf_df["F1"].apply(lambda x: f"{x:.4f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(clf_df, x="Model", y="ROC-AUC",
                          color="ROC-AUC", color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                          title="ROC-AUC — All Classifiers",
                          text=clf_df["ROC-AUC"].apply(lambda x: f"{x:.4f}"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=370, xaxis_tickangle=-25)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)
        bm     = S["clf_models"][best_clf]
        use_sc = best_clf in ["Logistic Regression","SVM (Linear)","KNN"]
        Xte_c  = S["Xte_c_sc"] if use_sc else S["X_test_c"]
        preds_c = bm.predict(Xte_c)
        cm      = confusion_matrix(yc_test, preds_c)

        with col3:
            sec(f"🔢 Confusion Matrix — {best_clf}")
            fig3, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                        xticklabels=["Low","High"],
                        yticklabels=["Low","High"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {best_clf}")
            plt.tight_layout(); st.pyplot(fig3); plt.close()

        with col4:
            sec(f"📈 ROC Curve — {best_clf}")
            if hasattr(bm,"predict_proba"):
                proba_c = bm.predict_proba(Xte_c)[:,1]
                fpr,tpr,_ = roc_curve(yc_test, proba_c)
                auc_val   = roc_auc_score(yc_test, proba_c)
                fig4, ax2 = plt.subplots(figsize=(5,4))
                ax2.plot(fpr, tpr, color=CLR["amber"], lw=2.5, label=f"AUC={auc_val:.4f}")
                ax2.plot([0,1],[0,1], color=CLR["grey"], ls="--")
                ax2.fill_between(fpr, tpr, alpha=0.15, color=CLR["amber"])
                ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
                ax2.set_title(f"ROC Curve — {best_clf}"); ax2.legend()
                plt.tight_layout(); st.pyplot(fig4); plt.close()

        insight(f"Best: {best_clf} — strong performance expected (R² + temporal features).")
        info("Since balance is ~50/50, accuracy is also a valid metric here alongside F1 and AUC.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔑 Tab 4 — Feature Importance")

    if not S.get("clf_models"):
        warn("Train at least one model in Tab 1.")
    else:
        clf_df = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        reg_df = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True)
        feats  = S["X_cols"]

        col1, col2 = st.columns(2)
        with col1:
            sec("🎯 Classification Importance")
            best_clf = clf_df.iloc[0]["Model"]
            bm = S["clf_models"][best_clf]
            if hasattr(bm,"feature_importances_"):
                imp = pd.DataFrame({"Feature":feats,"Importance":bm.feature_importances_})\
                        .sort_values("Importance",ascending=True)
                fig, ax = plt.subplots(figsize=(7,max(5,len(imp)*0.33)))
                colors_i = [CLR["amber"] if i>=len(imp)-3 else CLR["primary"]
                            for i in range(len(imp))]
                ax.barh(imp["Feature"], imp["Importance"], color=colors_i)
                ax.set_xlabel("Importance"); ax.set_title(f"{best_clf}")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            elif hasattr(bm,"coef_"):
                coef = pd.DataFrame({"Feature":feats,
                                     "Coef":np.abs(bm.coef_[0] if bm.coef_.ndim>1 else bm.coef_)})\
                         .sort_values("Coef",ascending=True)
                fig, ax = plt.subplots(figsize=(7,max(5,len(coef)*0.33)))
                ax.barh(coef["Feature"], coef["Coef"], color=CLR["amber"])
                ax.set_xlabel("|Coefficient|"); ax.set_title(f"{best_clf}")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                info(f"{best_clf} doesn't expose importances.")

        with col2:
            sec("📈 Regression Importance")
            if not reg_df.empty:
                best_reg = reg_df.iloc[0]["Model"]
                rm = S["reg_models"][best_reg]
                if hasattr(rm,"feature_importances_"):
                    imp2 = pd.DataFrame({"Feature":feats,"Importance":rm.feature_importances_})\
                             .sort_values("Importance",ascending=True)
                    fig2, ax2 = plt.subplots(figsize=(7,max(5,len(imp2)*0.33)))
                    ax2.barh(imp2["Feature"], imp2["Importance"], color=CLR["teal"])
                    ax2.set_xlabel("Importance"); ax2.set_title(f"{best_reg}")
                    plt.tight_layout(); st.pyplot(fig2); plt.close()
                elif hasattr(rm,"coef_"):
                    coef2 = pd.DataFrame({"Feature":feats,"Coef":np.abs(rm.coef_)})\
                              .sort_values("Coef",ascending=True)
                    fig2, ax2 = plt.subplots(figsize=(7,max(5,len(coef2)*0.33)))
                    ax2.barh(coef2["Feature"], coef2["Coef"], color=CLR["teal"])
                    ax2.set_xlabel("|Coef|"); ax2.set_title(f"{best_reg}")
                    plt.tight_layout(); st.pyplot(fig2); plt.close()
                else:
                    info(f"{best_reg} doesn't expose importances.")
            else:
                info("Train regression models in Tab 1.")

        insight("Rolling_24h_mean and Global_intensity typically dominate — temporal context is king.")
        warn("Global_intensity is physically derived from power — high importance is expected but reduces interpretability.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — PREDICT
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("🔮 Tab 5 — Interactive Energy Consumption Prediction")

    if not S.get("clf_models"):
        warn("Train at least one model in Tab 1.")
    else:
        clf_df = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        info("Enter current conditions to predict next-hour consumption.")

        col1, col2, col3 = st.columns(3)
        with col1:
            sec("⏰ Time Context")
            hour       = st.slider("Hour of Day", 0, 23, 18)
            dow        = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
            dow_map    = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
            month      = st.slider("Month", 1, 12, 1)
            weekend    = 1 if dow in ["Sat","Sun"] else 0
            is_peak    = 1 if hour in [7,8,9,18,19,20,21] else 0

        with col2:
            sec("🔌 Current Readings")
            reactive   = st.number_input("Global Reactive Power (kW)", 0.0, 1.5, 0.1, 0.01)
            voltage    = st.number_input("Voltage (V)", 220.0, 255.0, 240.0, 0.5)
            intensity  = st.number_input("Global Intensity (A)", 0.0, 50.0, 8.0, 0.5)
            sub1       = st.number_input("Sub_metering_1 Kitchen (Wh)", 0.0, 80.0, 5.0, 1.0)
            sub2       = st.number_input("Sub_metering_2 Laundry (Wh)", 0.0, 80.0, 1.0, 1.0)
            sub3       = st.number_input("Sub_metering_3 HVAC (Wh)",    0.0, 32.0, 17.0, 1.0)

        with col3:
            sec("📊 Computed Values")
            total_sub  = sub1 + sub2 + sub3
            active_est = intensity * voltage / 1000
            unmetered  = max(0, active_est*1000/60 - total_sub)
            pf_est     = active_est / (active_est + reactive + 1e-6)
            roll_24h   = float(df["Rolling_24h_mean"].mean())
            roll_7d    = float(df["Rolling_7d_mean"].mean())

            st.metric("Total Sub-metering", f"{total_sub:.1f} Wh")
            st.metric("Est. Active Power",  f"{active_est:.3f} kW")
            st.metric("Unmetered",          f"{unmetered:.1f} Wh")
            st.metric("Power Factor",       f"{pf_est:.3f}")
            st.metric("Peak Hour",          "🔴 Yes" if is_peak else "🟢 No")

        st.markdown("---")
        if st.button("🔮 Predict Consumption", type="primary", use_container_width=True):
            row_dict = {
                "Global_reactive_power": reactive,
                "Voltage":               voltage,
                "Global_intensity":      intensity,
                "Sub_metering_1":        sub1,
                "Sub_metering_2":        sub2,
                "Sub_metering_3":        sub3,
                "Total_submetering":     total_sub,
                "Unmetered_power":       unmetered,
                "Power_factor":          pf_est,
                "Hour":                  hour,
                "DayOfWeek":             dow_map[dow],
                "Month":                 month,
                "Weekend":               weekend,
                "is_peak":               is_peak,
                "Rolling_24h_mean":      roll_24h,
                "Rolling_7d_mean":       roll_7d,
            }
            input_row = pd.DataFrame([{k: row_dict.get(k,0) for k in S["X_cols"]}])
            input_sc  = S["scaler"].transform(input_row)

            # Regression prediction
            if S.get("reg_models"):
                reg_df2 = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False)
                best_r  = reg_df2.iloc[0]["Model"]
                rm = S["reg_models"][best_r]
                Xin_r = input_sc if best_r in ["Linear Regression","Ridge","Lasso"] else input_row
                pred_kw = rm.predict(Xin_r)[0]
                sec(f"⚡ Predicted Power — {best_r}")
                st.metric("Predicted Global Active Power", f"{pred_kw:.4f} kW",
                          delta=f"{pred_kw - df[REG_TARGET].mean():+.4f} vs avg")

            # Classification
            sec("🎯 High Consumption? — All Classifiers")
            pred_rows = []
            for name, model in S["clf_models"].items():
                use_sc = name in ["Logistic Regression","SVM (Linear)","KNN"]
                Xin    = input_sc if use_sc else input_row
                pred   = model.predict(Xin)[0]
                prob   = model.predict_proba(Xin)[0][1] if hasattr(model,"predict_proba") else None
                pred_rows.append({
                    "Model":       name,
                    "Prediction":  "🔴 HIGH" if pred==1 else "🟢 LOW",
                    "Probability": f"{prob*100:.1f}%" if prob is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)

            high_votes = sum(1 for r in pred_rows if "HIGH" in r["Prediction"])
            verdict    = "🔴 HIGH CONSUMPTION" if high_votes > len(pred_rows)/2 else "🟢 LOW CONSUMPTION"
            if "HIGH" in verdict:
                st.error(f"{verdict} — {high_votes}/{len(pred_rows)} models predict high consumption.")
            else:
                st.success(f"{verdict} — only {high_votes}/{len(pred_rows)} models predict high consumption.")
