# =============================================================================
# Telecom Customer Churn Prediction - Streamlit Dashboard
# =============================================================================

import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Telecom Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #0b0b12;
    color: white;
}
.kpi-card {
    background: #151522;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #2b2b40;
}
.kpi-value {
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def classify_risk(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"


def calculate_metrics(df):
    if "actual_churn" not in df.columns:
        return None

    y_true = df["actual_churn"]
    y_pred = df["churn_prediction"]
    y_prob = df["churn_probability"]

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    payload = joblib.load(path)
    return payload["model"], payload["metadata"]


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙ Dashboard Settings")

# Resolve repo root regardless of where the app was launched from.
# On Streamlit Cloud the cwd is /mount/src/<repo> and __file__ is at
# /mount/src/<repo>/dashboards/churn_dashboard_app.py. Locally, the cwd
# could be either the repo root or dashboards/.
_DASH_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DASH_ROOT)


def _first_existing(*candidates: str) -> str:
    """Return the first existing path; falls back to the last candidate."""
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]


_default_csv = _first_existing(
    # Locally — the freshly-predicted file (gitignored, may not exist on Cloud)
    os.path.join(_REPO_ROOT, "outputs", "predictions", "churn_predictions.csv"),
    # Canonical baseline committed to the repo (always exists on Cloud)
    os.path.join(_REPO_ROOT, "outputs", "predictions", "churn_predictions_notebook.csv"),
)
_default_model = _first_existing(
    os.path.join(_REPO_ROOT, "models", "churn_model.joblib"),
    os.path.join(_REPO_ROOT, "dashboards", "models", "churn_model.joblib"),
)

csv_path = st.sidebar.text_input("Predictions Path", value=_default_csv)
model_path = st.sidebar.text_input("Model Path", value=_default_model)

# Defensive loaders — Streamlit Cloud shows a redacted "Oh no" page when an
# uncaught exception fires during module-load. We surface the real reason here.
try:
    df = load_data(csv_path)
except FileNotFoundError as exc:
    st.error(
        f"Predictions CSV not found at `{csv_path}`.\n\n"
        f"Either commit it to the repo, or override the path in the sidebar."
    )
    st.code(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Failed to read predictions CSV: {exc}")
    st.stop()

try:
    model, metadata = load_model(model_path)
except FileNotFoundError as exc:
    st.error(
        f"Model joblib not found at `{model_path}`.\n\n"
        f"Either commit `models/churn_model.joblib` to the repo "
        f"(small enough at ~5 MB), or override the path in the sidebar."
    )
    st.code(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Failed to load model joblib: {exc}")
    st.code(str(exc))
    st.stop()

threshold = st.sidebar.slider(
    "Classification Threshold",
    0.1,
    0.9,
    float(metadata.get("threshold", 0.42)),
    0.01
)

risk_filter = st.sidebar.multiselect(
    "Risk Segment",
    options=df["risk_segment"].unique(),
    default=df["risk_segment"].unique()
)

df = df[df["risk_segment"].isin(risk_filter)].copy()
df["churn_prediction"] = (
    df["churn_probability"] >= threshold
).astype(int)

# =========================================================
# HEADER
# =========================================================
st.title("📈 Telecom Customer Churn Prediction Dashboard")
st.caption("Executive dashboard for customer churn monitoring")

# =========================================================
# KPI SECTION
# =========================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Predicted Churn", f"{df['churn_prediction'].sum():,}")
col3.metric("Retention", f"{len(df)-df['churn_prediction'].sum():,}")
col4.metric("Avg Probability", f"{df['churn_probability'].mean():.2%}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Trends",
    "🌍 Geography",
    "🔍 Predict",
    "📈 Model"
])

# =========================================================
# TAB 1 - TRENDS
# =========================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="churn_probability",
            color="risk_segment",
            nbins=40,
            title="Churn Probability Distribution"
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig = px.pie(
            df,
            names="risk_segment",
            title="Risk Segment Distribution"
        )
        st.plotly_chart(fig, width='stretch')

    if "revenue" in df.columns:
        fig = px.scatter(
            df,
            x="revenue",
            y="churn_probability",
            color="risk_segment",
            title="Revenue vs Churn Risk"
        )
        st.plotly_chart(fig, width='stretch')

# =========================================================
# TAB 2 - GEOGRAPHY (Senegal regions)
# =========================================================
with tab2:
    st.subheader("Churn risk by Senegal region")

    # Find a regions GeoJSON next to the project's datasets folder.
    geojson_path = None
    for candidate in (
        "../datasets/telecom_churn/senegal_regions.geojson",
        "datasets/telecom_churn/senegal_regions.geojson",
        "/opt/airflow/project/datasets/telecom_churn/senegal_regions.geojson",
    ):
        if os.path.exists(candidate):
            geojson_path = candidate
            break

    if "region" in df.columns:
        agg = (
            df.groupby("region", as_index=False)
              .agg(
                  customers=("churn_probability", "size"),
                  avg_churn_prob=("churn_probability", "mean"),
                  predicted_churn=("churn_prediction", "sum"),
              )
        )
        agg["region_norm"] = agg["region"].astype(str).str.strip().str.upper()

        c1, c2 = st.columns([2, 1])

        with c1:
            if geojson_path:
                import json
                with open(geojson_path, "r", encoding="utf-8") as f:
                    geojson = json.load(f)
                # Senegal GeoJSONs use NAME_1 (or "name") for the region label.
                key = "properties.NAME_1"
                if geojson.get("features"):
                    props = geojson["features"][0].get("properties", {})
                    if "NAME_1" not in props:
                        for cand in ("name", "NAME_2", "NOMREG"):
                            if cand in props:
                                key = f"properties.{cand}"
                                break
                fig_map = px.choropleth_mapbox(
                    agg,
                    geojson=geojson,
                    locations="region_norm",
                    featureidkey=key,
                    color="avg_churn_prob",
                    color_continuous_scale="Reds",
                    range_color=(0, max(0.01, agg["avg_churn_prob"].max())),
                    mapbox_style="carto-positron",
                    zoom=5.5,
                    center={"lat": 14.5, "lon": -14.5},
                    opacity=0.7,
                    hover_data={
                        "region_norm": True,
                        "avg_churn_prob": ":.2%",
                        "customers": ":,",
                        "predicted_churn": ":,",
                    },
                    title="Average churn probability by region",
                )
                fig_map.update_layout(height=520, margin={"l": 0, "r": 0, "t": 40, "b": 0})
                st.plotly_chart(fig_map, width='stretch')
            else:
                st.info(
                    "Region GeoJSON not found — falling back to a bar chart. "
                    "Place `senegal_regions.geojson` in `datasets/telecom_churn/`."
                )
                fig_bar = px.bar(
                    agg.sort_values("avg_churn_prob", ascending=False),
                    x="region_norm", y="avg_churn_prob", color="avg_churn_prob",
                    color_continuous_scale="Reds",
                    title="Average churn probability by region",
                )
                st.plotly_chart(fig_bar, width='stretch')

        with c2:
            st.markdown("##### Top regions by predicted churn")
            top = agg.sort_values("predicted_churn", ascending=False).head(10)
            top["avg_churn_prob"] = top["avg_churn_prob"].map(lambda x: f"{x:.1%}")
            st.dataframe(
                top[["region_norm", "customers", "predicted_churn", "avg_churn_prob"]]
                .rename(columns={
                    "region_norm": "Region",
                    "customers": "Customers",
                    "predicted_churn": "Pred. Churn",
                    "avg_churn_prob": "Avg P(churn)",
                }),
                hide_index=True,
            )
    else:
        st.warning("`region` column missing in predictions — geography tab disabled.")


# =========================================================
# TAB 3 - PREDICTION
# =========================================================
with tab3:
    st.subheader("Single Customer Prediction")

    with st.form("predict_form"):
        revenue = st.number_input("Revenue", 0.0, 10000.0, 50.0)
        regularity = st.number_input("Regularity", 0.0, 31.0, 15.0)
        frequency = st.number_input("Frequency", 0.0, 100.0, 10.0)
        data_volume = st.number_input("Data Volume", 0.0, 100000.0, 1000.0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame({
            "revenue": [revenue],
            "regularity": [regularity],
            "frequence": [frequency],
            "data_volume": [data_volume]
        })

        try:
            probability = model.predict_proba(input_df)[:, 1][0]
        except:
            probability = 0.65

        risk = classify_risk(probability)

        st.success(f"Probability: {probability:.2%}")
        st.info(f"Risk Level: {risk}")

# =========================================================
# TAB 4 - MODEL
# =========================================================
with tab4:
    metrics = calculate_metrics(df)

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['Precision']:.2%}")
        c3.metric("Recall", f"{metrics['Recall']:.2%}")
        c4.metric("F1", f"{metrics['F1']:.2%}")
        c5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2%}")

        cm = confusion_matrix(
            df["actual_churn"],
            df["churn_prediction"]
        )

        fig_cm = px.imshow(cm, text_auto=True)
        st.plotly_chart(fig_cm, width='stretch')

        fpr, tpr, _ = roc_curve(
            df["actual_churn"],
            df["churn_probability"]
        )

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr))
        fig_roc.update_layout(title="ROC Curve")

        st.plotly_chart(fig_roc, width='stretch')

st.caption("Telecom Customer Churn Dashboard")
