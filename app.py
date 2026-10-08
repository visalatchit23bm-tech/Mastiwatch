
# ============================================================
# MASTIWATCH
# AI-BASED BOVINE MASTITIS EARLY WARNING SYSTEM
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
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
    precision_recall_curve,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MastiWatch | AI Mastitis Early Warning",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5faf7;
    }

    .main-header {
        background: linear-gradient(135deg, #14532d, #166534);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.10);
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
    }

    .main-subtitle {
        font-size: 18px;
        margin-top: 8px;
        opacity: 0.95;
    }

    .main-description {
        font-size: 14px;
        margin-top: 12px;
        opacity: 0.85;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }

    .metric-title {
        color: #6b7280;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #14532d;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-title {
        color: #14532d;
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 18px;
    }

    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    .info-box {
        background: white;
        border-left: 5px solid #16a34a;
        padding: 18px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    }

    .footer-box {
        text-align: center;
        padding: 25px;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "mastitis_model.pkl"
DATA_PATH = BASE_DIR / "mastitis_features.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(
            "❌ mastitis_model.pkl was not found.\n\n"
            "Make sure it is in the same folder as app.py."
        )
        st.stop()

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

    if isinstance(bundle, dict):

        if "model" not in bundle:
            st.error(
                "❌ The model file does not contain a 'model' object."
            )
            st.stop()

        model = bundle["model"]

        if "features" in bundle:
            features = list(bundle["features"])

        elif hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)

        else:
            st.error(
                "❌ Feature names could not be found in the model."
            )
            st.stop()

    else:

        model = bundle

        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        else:
            st.error(
                "❌ Your model does not contain feature names."
            )
            st.stop()

    return model, features


MODEL, FEATURES = load_model()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error(
            "❌ mastitis_features.csv was not found.\n\n"
            "Make sure it is in the same folder as app.py."
        )
        st.stop()

    try:
        data = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"❌ CSV loading failed: {e}")
        st.stop()

    return data


DF = load_data()


# ============================================================
# CHECK MODEL FEATURES
# ============================================================

missing_features = [
    feature for feature in FEATURES
    if feature not in DF.columns
]

if missing_features:

    st.error("❌ Model features missing from CSV:")

    for feature in missing_features:
        st.write(f"- `{feature}`")

    st.stop()


# ============================================================
# CALCULATE RISK
# ============================================================

@st.cache_data
def calculate_risk(data):

    result = data.copy()

    try:
        result["risk"] = MODEL.predict_proba(
            result[FEATURES]
        )[:, 1]

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    return result


DF = calculate_risk(DF)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_band(
    risk,
    high_threshold=0.70,
    medium_threshold=0.40
):

    if risk >= high_threshold:
        return "High"

    elif risk >= medium_threshold:
        return "Medium"

    return "Low"


def get_risk_class(band):

    if band == "High":
        return "risk-high"

    elif band == "Medium":
        return "risk-medium"

    return "risk-low"


def get_id_column(data):

    possible_columns = [
        "cow_id",
        "id",
        "cow",
        "animal_id",
    ]

    for column in possible_columns:

        if column in data.columns:
            return column

    return None


# ============================================================
# HEADER
# IMPORTANT: NO BLANK LINES INSIDE HTML
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">🐄 MastiWatch</div>
        <div class="main-subtitle">AI-Based Early Warning System for Bovine Mastitis</div>
        <div class="main-description">Intelligent herd monitoring • Risk prediction • Early intervention</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🐄 MastiWatch")

    st.caption(
        "AI-powered bovine mastitis early warning platform"
    )

    st.divider()

    st.markdown("### ⚙️ Risk Thresholds")

    high_threshold = st.slider(
        "High Risk",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
    )

    medium_threshold = st.slider(
        "Medium Risk",
        min_value=0.20,
        max_value=0.65,
        value=0.40,
        step=0.05,
    )

    if medium_threshold >= high_threshold:
        medium_threshold = high_threshold - 0.05

    st.divider()

    st.markdown("### 🌐 Language")

    language = st.selectbox(
        "Language",
        ["English", "Tamil"]
    )

    st.divider()

    st.markdown("### 📊 Dataset")

    st.write(f"**Records:** {len(DF):,}")
    st.write(f"**AI Features:** {len(FEATURES)}")

    st.divider()

    st.caption(
        "Research prototype. AI risk prediction does not "
        "replace professional veterinary diagnosis."
    )


# ============================================================
# CREATE RISK BANDS
# ============================================================

DF["risk_band"] = DF["risk"].apply(
    lambda x: get_risk_band(
        x,
        high_threshold,
        medium_threshold
    )
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Herd Dashboard",
        "🐄 Cow Intelligence",
        "🧪 What-If Simulator",
        "📈 AI Performance",
        "ℹ️ About",
    ]
)


# ============================================================
# TAB 1 — HERD DASHBOARD
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">Herd Risk Overview</div>',
        unsafe_allow_html=True
    )

    total_cows = len(DF)

    high_count = int(
        (DF["risk"] >= high_threshold).sum()
    )

    medium_count = int(
        (
            (DF["risk"] >= medium_threshold)
            &
            (DF["risk"] < high_threshold)
        ).sum()
    )

    low_count = int(
        (DF["risk"] < medium_threshold).sum()
    )

    average_risk = DF["risk"].mean()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Cows</div>
                <div class="metric-value">{total_cows:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🔴 High Risk</div>
                <div class="metric-value">{high_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🟡 Medium Risk</div>
                <div class="metric-value">{medium_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🟢 Low Risk</div>
                <div class="metric-value">{low_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.markdown("### 🔎 Herd Filters")

    f1, f2 = st.columns(2)

    with f1:

        if "breed_code" in DF.columns:

            breed_options = sorted(
                DF["breed_code"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_breed = st.selectbox(
                "Breed",
                ["All"] + breed_options
            )

        else:

            selected_breed = "All"

    with f2:

        selected_risk = st.selectbox(
            "Risk Level",
            ["All", "High", "Medium", "Low"]
        )

    filtered_df = DF.copy()

    if selected_breed != "All":

        filtered_df = filtered_df[
            filtered_df["breed_code"] == selected_breed
        ]

    if selected_risk != "All":

        filtered_df = filtered_df[
            filtered_df["risk_band"] == selected_risk
        ]

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    chart1, chart2 = st.columns(2)

    with chart1:

        risk_counts = (
            filtered_df["risk_band"]
            .value_counts()
            .reindex(
                ["High", "Medium", "Low"],
                fill_value=0
            )
        )

        pie_df = pd.DataFrame(
            {
                "Risk Level": risk_counts.index,
                "Cows": risk_counts.values
            }
        )

        fig = px.pie(
            pie_df,
            names="Risk Level",
            values="Cows",
            hole=0.55,
            title="Herd Risk Distribution"
        )

        fig.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=60,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with chart2:

        if "breed_code" in filtered_df.columns:

            breed_risk = (
                filtered_df
                .groupby("breed_code")["risk"]
                .mean()
                .reset_index()
                .sort_values(
                    "risk",
                    ascending=False
                )
            )

            breed_risk["risk_percent"] = (
                breed_risk["risk"] * 100
            )

            fig = px.bar(
                breed_risk,
                x="breed_code",
                y="risk_percent",
                title="Average Risk by Breed",
                labels={
                    "breed_code": "Breed",
                    "risk_percent": "Risk (%)"
                }
            )

            fig.update_layout(
                height=400,
                margin=dict(
                    l=10,
                    r=10,
                    t=60,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # PRIORITY COWS
    # --------------------------------------------------------

    st.markdown("### 🚨 Priority Cows")

    priority_df = (
        filtered_df
        .sort_values(
            "risk",
            ascending=False
        )
        .head(15)
        .copy()
    )

    id_column = get_id_column(priority_df)

    display_columns = []

    if id_column:
        display_columns.append(id_column)

    for column in [
        "breed_code",
        "scc",
        "yield_l",
        "body_temp",
        "conductivity",
        "humidity",
        "thi",
    ]:

        if column in priority_df.columns:
            display_columns.append(column)

    display_columns.append("risk")

    display_df = priority_df[
        display_columns
    ].copy()

    display_df["risk"] = (
        display_df["risk"] * 100
    ).round(1)

    display_df = display_df.rename(
        columns={
            "risk": "Risk (%)"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "⬇️ Download Herd Risk Report",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="MastiWatch_Herd_Risk_Report.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 2 — COW INTELLIGENCE
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">🐄 Individual Cow Intelligence</div>',
        unsafe_allow_html=True
    )

    id_column = get_id_column(DF)

    if id_column:

        cow_options = (
            DF[id_column]
            .dropna()
            .unique()
            .tolist()
        )

        selected_cow = st.selectbox(
            "Select Cow",
            cow_options,
            key="cow_select"
        )

        cow = DF[
            DF[id_column] == selected_cow
        ].iloc[0]

    else:

        selected_index = st.selectbox(
            "Select Cow Record",
            DF.index.tolist(),
            key="cow_select_index"
        )

        cow = DF.loc[selected_index]

    cow_risk = float(cow["risk"])

    cow_band = get_risk_band(
        cow_risk,
        high_threshold,
        medium_threshold
    )

    risk_css = get_risk_class(
        cow_band
    )

    left, right = st.columns(2)

    # --------------------------------------------------------
    # GAUGE
    # --------------------------------------------------------

    with left:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=cow_risk * 100,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 38
                    }
                },
                title={
                    "text": "Predicted Mastitis Risk"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": "#166534"
                    },
                    "steps": [
                        {
                            "range": [
                                0,
                                medium_threshold * 100
                            ],
                            "color": "#dcfce7"
                        },
                        {
                            "range": [
                                medium_threshold * 100,
                                high_threshold * 100
                            ],
                            "color": "#fef3c7"
                        },
                        {
                            "range": [
                                high_threshold * 100,
                                100
                            ],
                            "color": "#fee2e2"
                        }
                    ]
                }
            )
        )

        fig.update_layout(
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # COW DETAILS
    # --------------------------------------------------------

    with right:

        st.markdown("### 🐄 Cow Profile")

        st.markdown(
            f'<div class="{risk_css}">{cow_band} Risk</div>',
            unsafe_allow_html=True
        )

        st.write("")

        detail_columns = [
            "breed_code",
            "parity",
            "scc",
            "conductivity",
            "yield_l",
            "body_temp",
            "humidity",
            "thi",
            "hygiene",
            "hand_milking"
        ]

        details = []

        for column in detail_columns:

            if column in cow.index:

                value = cow[column]

                if pd.notna(value):

                    details.append(
                        {
                            "Parameter": column.replace(
                                "_",
                                " "
                            ).title(),
                            "Value": value
                        }
                    )

        if details:

            st.dataframe(
                pd.DataFrame(details),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # AI EXPLANATION
    # --------------------------------------------------------

    st.markdown("### 🧠 AI Risk Drivers")

    try:

        if hasattr(MODEL, "get_booster"):

            import xgboost as xgb

            row = cow[
                FEATURES
            ].to_frame().T

            matrix = xgb.DMatrix(
                row,
                feature_names=FEATURES
            )

            contributions = MODEL.get_booster().predict(
                matrix,
                pred_contribs=True
            )[0]

            contributions = contributions[:-1]

            contribution_series = pd.Series(
                contributions,
                index=FEATURES
            )

            contribution_df = (
                contribution_series
                .sort_values()
                .tail(10)
                .reset_index()
            )

            contribution_df.columns = [
                "Feature",
                "Contribution"
            ]

            fig = px.bar(
                contribution_df,
                x="Contribution",
                y="Feature",
                orientation="h",
                title="Top AI Risk Contributors"
            )

            fig.update_layout(
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "AI explanation is unavailable "
                "for this model type."
            )

    except Exception as e:

        st.info(
            f"AI explanation unavailable: {e}"
        )


# ============================================================
# TAB 3 — WHAT IF SIMULATOR
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">🧪 What-If Risk Simulator</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Change selected parameters to explore how the "
        "predicted risk may change. This is a simulation "
        "and not a veterinary diagnosis."
    )

    # Select cow
    if id_column:

        simulator_options = (
            DF[id_column]
            .dropna()
            .unique()
            .tolist()
        )

        simulator_cow = st.selectbox(
            "Select Reference Cow",
            simulator_options,
            key="simulator_select"
        )

        base_cow = DF[
            DF[id_column] == simulator_cow
        ].iloc[0].copy()

    else:

        simulator_index = st.selectbox(
            "Select Reference Cow",
            DF.index.tolist(),
            key="simulator_index"
        )

        base_cow = DF.loc[
            simulator_index
        ].copy()

    simulation = base_cow[
        FEATURES
    ].copy()

    editable_features = [
        feature
        for feature in [
            "scc",
            "conductivity",
            "yield_l",
            "body_temp",
            "humidity",
            "thi",
            "hygiene"
        ]
        if feature in FEATURES
    ]

    st.markdown("### Modify Parameters")

    input_columns = st.columns(2)

    for i, feature in enumerate(
        editable_features
    ):

        try:

            current_value = float(
                base_cow[feature]
            )

        except Exception:

            continue

        minimum = (
            current_value * 0.5
        )

        maximum = (
            current_value * 1.5
        )

        if current_value == 0:

            minimum = -1
            maximum = 1

        with input_columns[i % 2]:

            new_value = st.number_input(
                feature.replace(
                    "_",
                    " "
                ).title(),
                value=current_value,
                min_value=float(minimum),
                max_value=float(maximum),
                step=max(
                    abs(current_value) * 0.01,
                    0.01
                ),
                key=f"whatif_{feature}"
            )

            simulation[feature] = new_value

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        original_probability = float(
            MODEL.predict_proba(
                pd.DataFrame(
                    [base_cow[FEATURES]],
                    columns=FEATURES
                )
            )[0][1]
        )

        simulated_probability = float(
            MODEL.predict_proba(
                pd.DataFrame(
                    [simulation],
                    columns=FEATURES
                )
            )[0][1]
        )

        difference = (
            simulated_probability
            - original_probability
        )

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Original Risk",
                f"{original_probability * 100:.1f}%"
            )

        with m2:

            st.metric(
                "Simulated Risk",
                f"{simulated_probability * 100:.1f}%"
            )

        with m3:

            st.metric(
                "Risk Change",
                f"{difference * 100:+.1f}%"
            )

        simulated_band = get_risk_band(
            simulated_probability,
            high_threshold,
            medium_threshold
        )

        st.markdown(
            f"""
            <div class="info-box">
                <b>Simulation Result:</b><br>
                Predicted risk category:
                <b>{simulated_band}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            f"❌ What-if simulation failed: {e}"
        )


# ============================================================
# TAB 4 — AI PERFORMANCE
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">📈 AI Model Performance</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Performance metrics require a ground-truth label "
        "column in the dataset."
    )

    label_column = None

    possible_labels = [
        "label",
        "target",
        "mastitis",
        "mastitis_label",
        "diagnosis",
        "y"
    ]

    for column in possible_labels:

        if column in DF.columns:

            label_column = column
            break

    if label_column:

        y_true = DF[
            label_column
        ].copy()

        # -----------------------------------------------
        # STRING LABEL CONVERSION
        # -----------------------------------------------

        if y_true.dtype == "object":

            label_map = {
                "yes": 1,
                "no": 0,
                "positive": 1,
                "negative": 0,
                "mastitis": 1,
                "healthy": 0,
                "infected": 1,
                "normal": 0,
                "1": 1,
                "0": 0
            }

            y_true = (
                y_true
                .astype(str)
                .str.lower()
                .map(label_map)
            )

        y_true = pd.to_numeric(
            y_true,
            errors="coerce"
        )

        valid = y_true.notna()

        y_true = y_true[
            valid
        ].astype(int)

        y_probability = DF.loc[
            valid,
            "risk"
        ]

        y_prediction = (
            y_probability >= 0.50
        ).astype(int)

        if (
            len(y_true) > 0
            and y_true.nunique() == 2
        ):

            accuracy = accuracy_score(
                y_true,
                y_prediction
            )

            precision = precision_score(
                y_true,
                y_prediction,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_prediction,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_prediction,
                zero_division=0
            )

            auc = roc_auc_score(
                y_true,
                y_probability
            )

            # -------------------------------------------
            # METRICS
            # -------------------------------------------

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.1f}%"
                )

            with c2:
                st.metric(
                    "Precision",
                    f"{precision * 100:.1f}%"
                )

            with c3:
                st.metric(
                    "Recall",
                    f"{recall * 100:.1f}%"
                )

            with c4:
                st.metric(
                    "F1 Score",
                    f"{f1 * 100:.1f}%"
                )

            with c5:
                st.metric(
                    "ROC-AUC",
                    f"{auc:.3f}"
                )

            # -------------------------------------------
            # ROC CURVE
            # -------------------------------------------

            fpr, tpr, _ = roc_curve(
                y_true,
                y_probability
            )

            roc_df = pd.DataFrame(
                {
                    "False Positive Rate": fpr,
                    "True Positive Rate": tpr
                }
            )

            fig = px.line(
                roc_df,
                x="False Positive Rate",
                y="True Positive Rate",
                title=f"ROC Curve — AUC = {auc:.3f}"
            )

            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    dash="dash"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -------------------------------------------
            # PRECISION RECALL CURVE
            # -------------------------------------------

            precision_curve, recall_curve, _ = (
                precision_recall_curve(
                    y_true,
                    y_probability
                )
            )

            pr_df = pd.DataFrame(
                {
                    "Recall": recall_curve,
                    "Precision": precision_curve
                }
            )

            fig = px.line(
                pr_df,
                x="Recall",
                y="Precision",
                title="Precision–Recall Curve"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -------------------------------------------
            # CONFUSION MATRIX
            # -------------------------------------------

            cm = confusion_matrix(
                y_true,
                y_prediction
            )

            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual Healthy",
                    "Actual Mastitis"
                ],
                columns=[
                    "Predicted Healthy",
                    "Predicted Mastitis"
                ]
            )

            fig = px.imshow(
                cm_df,
                text_auto=True,
                title="Confusion Matrix"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "The detected label column does not contain "
                "valid binary ground-truth values."
            )

    else:

        st.warning(
            "No ground-truth label column was found in "
            "mastitis_features.csv."
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown("### 🔍 Feature Importance")

    try:

        if hasattr(
            MODEL,
            "feature_importances_"
        ):

            importance_df = pd.DataFrame(
                {
                    "Feature": FEATURES,
                    "Importance":
                        MODEL.feature_importances_
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(15)
            )

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Model Features"
            )

            fig.update_layout(
                height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Feature importance is not available "
                "for this model."
            )

    except Exception as e:

        st.info(
            f"Feature importance unavailable: {e}"
        )


# ============================================================
# TAB 5 — ABOUT
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-title">ℹ️ About MastiWatch</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
            <h3>🐄 What is MastiWatch?</h3>
            MastiWatch is an AI-based early warning platform
            designed to identify cows that may have an increased
            risk of bovine mastitis.
            <br><br>
            The system analyzes cow-level, milk-quality,
            environmental and management-related indicators
            to estimate mastitis risk.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🤖 AI Pipeline")

        st.markdown(
            """
            **1. Data Collection**  
            Cow health, milk quality, environmental and
            management parameters.

            **2. Feature Engineering**  
            Moving averages, deviations and changes.

            **3. Machine Learning**  
            XGBoost-based classification.

            **4. Risk Prediction**  
            Probability score for each cow.

            **5. Early Warning**  
            Low, Medium and High risk categories.
            """
        )

    with col2:

        st.markdown("### 🚨 Recommended Action")

        st.markdown(
            """
            🟢 **Low Risk**  
            Continue routine monitoring.

            🟡 **Medium Risk**  
            Increase observation and monitoring.

            🔴 **High Risk**  
            Prioritize veterinary inspection
            and appropriate follow-up.
            """
        )

    st.divider()

    st.markdown("### 👩‍💻 Team — Med Sphere")

    team_df = pd.DataFrame(
        {
            "Team Member": [
                "Asmiyanaseem S",
                "Rakshita M",
                "Sherifa Beevi N",
                "Srirammuthiah C"
            ],
            "Institution": [
                "PSNA College of Engineering and Technology",
                "PSNA College of Engineering and Technology",
                "PSNA College of Engineering and Technology",
                "PSNA College of Engineering and Technology"
            ]
        }
    )

    st.dataframe(
        team_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown(
        """
        <div class="footer-box">
            <b>🐄 MastiWatch</b><br>
            AI-Based Bovine Mastitis Early Warning System
            <br><br>
            Built for preventive monitoring and
            smarter livestock healthcare.
            <br><br>
            ⚠️ Research Prototype — Not a Veterinary Diagnostic Tool
        </div>
        """,
        unsafe_allow_html=True
    )
