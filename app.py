# ============================================================
# MASTIWATCH
# AI-BASED BOVINE MASTITIS EARLY WARNING SYSTEM
# Team: Med Sphere
# PSNA College of Engineering and Technology
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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "mastitis_model.pkl"
DATA_PATH = BASE_DIR / "mastitis_features.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #f6faf7;
    }

    /* Header */
    .main-header {
        padding: 25px 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #14532d, #166534);
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.10);
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .main-subtitle {
        font-size: 16px;
        opacity: 0.90;
    }

    /* Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        min-height: 125px;
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

    .risk-high {
        background: #fee2e2;
        color: #991b1b;
        padding: 7px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-medium {
        background: #fef3c7;
        color: #92400e;
        padding: 7px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-low {
        background: #dcfce7;
        color: #166534;
        padding: 7px 14px;
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

    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #14532d;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(
            f"❌ Model file not found:\n\n"
            f"`{MODEL_PATH.name}`\n\n"
            f"Place the model file in the same folder as `app.py`."
        )
        st.stop()

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Unable to load model: {e}")
        st.stop()

    # Expected structure:
    # {"model": model, "features": features}

    if isinstance(bundle, dict):

        if "model" in bundle:
            model = bundle["model"]
        else:
            st.error("❌ `mastitis_model.pkl` does not contain a `model` object.")
            st.stop()

        if "features" in bundle:
            features = bundle["features"]
        else:
            st.error("❌ `mastitis_model.pkl` does not contain `features`.")
            st.stop()

    else:
        # Fallback if only the model was saved
        model = bundle

        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        else:
            st.error(
                "❌ Could not determine the model feature names.\n\n"
                "Your `.pkl` should contain `model` and `features`."
            )
            st.stop()

    return model, list(features)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error(
            f"❌ Dataset not found:\n\n"
            f"`{DATA_PATH.name}`\n\n"
            f"Place the CSV file in the same folder as `app.py`."
        )
        st.stop()

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"❌ Unable to read dataset: {e}")
        st.stop()

    return df


MODEL, FEATURES = load_model()
DF = load_data()


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature for feature in FEATURES
    if feature not in DF.columns
]

if missing_features:

    st.error(
        "❌ Model features are missing from the dataset:\n\n"
        + "\n".join([f"- {x}" for x in missing_features])
    )

    st.stop()


# ============================================================
# PREDICT RISK
# ============================================================

@st.cache_data
def calculate_risk(data):

    temp = data.copy()

    try:
        probabilities = MODEL.predict_proba(temp[FEATURES])[:, 1]
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    temp["risk"] = probabilities

    return temp


DF = calculate_risk(DF)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def risk_band(risk, high_threshold=0.70, medium_threshold=0.40):

    if risk >= high_threshold:
        return "High"

    if risk >= medium_threshold:
        return "Medium"

    return "Low"


def risk_class(risk, high_threshold=0.70, medium_threshold=0.40):

    band = risk_band(
        risk,
        high_threshold,
        medium_threshold
    )

    if band == "High":
        return "risk-high"

    if band == "Medium":
        return "risk-medium"

    return "risk-low"


def format_risk(risk):

    return f"{risk * 100:.1f}%"


def safe_value(row, column, default="N/A"):

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">

        <div class="main-title">
            🐄 MastiWatch
        </div>

        <div class="main-subtitle">
            AI-Based Early Warning System for Bovine Mastitis
        </div>

        <div style="margin-top:10px;font-size:14px;">
            Intelligent herd monitoring • Risk prediction • Early intervention
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🐄 MastiWatch")

    st.caption("AI-powered bovine mastitis monitoring")

    st.divider()

    st.markdown("### ⚙️ Risk Settings")

    high_threshold = st.slider(
        "High-risk threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
    )

    medium_threshold = st.slider(
        "Medium-risk threshold",
        min_value=0.20,
        max_value=0.70,
        value=0.40,
        step=0.05,
    )

    if medium_threshold >= high_threshold:
        medium_threshold = high_threshold - 0.05

    st.divider()

    st.markdown("### 🌐 Language")

    language = st.selectbox(
        "Select language",
        ["English", "Tamil"]
    )

    st.divider()

    st.markdown("### 📊 Dataset")

    st.write(f"Rows: **{len(DF):,}**")
    st.write(f"Features: **{len(FEATURES)}**")

    st.divider()

    st.caption(
        "MastiWatch is a research prototype and "
        "should not replace veterinary diagnosis."
    )


# ============================================================
# APPLY RISK BANDS
# ============================================================

DF["risk_band"] = DF["risk"].apply(
    lambda x: risk_band(
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
        unsafe_allow_html=True,
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

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Cows</div>
                <div class="metric-value">{total_cows:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🔴 High Risk</div>
                <div class="metric-value">{high_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🟡 Medium Risk</div>
                <div class="metric-value">{medium_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Average Risk</div>
                <div class="metric-value">
                    {average_risk * 100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.markdown("### 🔎 Herd Filters")

    f1, f2 = st.columns(2)

    with f1:

        if "breed_code" in DF.columns:

            breeds = sorted(
                DF["breed_code"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_breed = st.selectbox(
                "Breed",
                ["All"] + breeds
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

    col1, col2 = st.columns(2)

    with col1:

        risk_counts = filtered_df["risk_band"].value_counts()

        pie_df = pd.DataFrame(
            {
                "Risk Level": risk_counts.index,
                "Cows": risk_counts.values,
            }
        )

        fig = px.pie(
            pie_df,
            names="Risk Level",
            values="Cows",
            title="Herd Risk Distribution",
            hole=0.55,
        )

        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        if "breed_code" in filtered_df.columns:

            breed_risk = (
                filtered_df
                .groupby("breed_code")["risk"]
                .mean()
                .reset_index()
                .sort_values("risk", ascending=False)
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
                    "risk_percent": "Risk (%)",
                },
            )

            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=50, b=10),
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # PRIORITY COWS
    # --------------------------------------------------------

    st.markdown(
        "### 🚨 Priority Cows"
    )

    priority = (
        filtered_df
        .sort_values("risk", ascending=False)
        .head(15)
        .copy()
    )

    display_columns = []

    for column in [
        "cow_id",
        "id",
        "cow",
        "breed_code",
        "scc",
        "yield_l",
        "body_temp",
        "conductivity",
    ]:

        if column in priority.columns:
            display_columns.append(column)

    display_columns.append("risk")

    priority_display = priority[display_columns].copy()

    priority_display["risk"] = (
        priority_display["risk"] * 100
    ).round(1)

    priority_display = priority_display.rename(
        columns={"risk": "Risk (%)"}
    )

    st.dataframe(
        priority_display,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Herd Risk Report",
        data=csv_data,
        file_name="mastiwatch_herd_risk_report.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 2 — COW INTELLIGENCE
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">Individual Cow Intelligence</div>',
        unsafe_allow_html=True,
    )

    # Select cow
    possible_id_columns = [
        "cow_id",
        "id",
        "cow",
    ]

    id_column = None

    for col in possible_id_columns:

        if col in DF.columns:
            id_column = col
            break

    if id_column is None:

        cow_options = list(DF.index)

        selected_cow = st.selectbox(
            "Select Cow",
            cow_options
        )

        cow = DF.loc[selected_cow]

    else:

        cow_options = (
            DF[id_column]
            .dropna()
            .unique()
            .tolist()
        )

        selected_cow = st.selectbox(
            "Select Cow",
            cow_options
        )

        cow = DF[
            DF[id_column] == selected_cow
        ].iloc[0]

    risk = float(cow["risk"])

    band = risk_band(
        risk,
        high_threshold,
        medium_threshold
    )

    css_class = risk_class(
        risk,
        high_threshold,
        medium_threshold
    )

    col1, col2 = st.columns([1, 1])

    # --------------------------------------------------------
    # RISK GAUGE
    # --------------------------------------------------------

    with col1:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                number={
                    "suffix": "%",
                    "font": {"size": 38},
                },
                title={
                    "text": "Mastitis Risk"
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
                            "range": [0, medium_threshold * 100],
                            "color": "#dcfce7",
                        },
                        {
                            "range": [
                                medium_threshold * 100,
                                high_threshold * 100,
                            ],
                            "color": "#fef3c7",
                        },
                        {
                            "range": [
                                high_threshold * 100,
                                100,
                            ],
                            "color": "#fee2e2",
                        },
                    ],
                },
            )
        )

        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=70, b=20),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # COW DETAILS
    # --------------------------------------------------------

    with col2:

        st.markdown("### 🐄 Cow Profile")

        st.markdown(
            f"""
            <div class="{css_class}">
                {band} Risk
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        details = {}

        for col in [
            "breed_code",
            "parity",
            "scc",
            "conductivity",
            "yield_l",
            "body_temp",
            "humidity",
            "thi",
            "hygiene",
            "hand_milking",
        ]:

            if col in cow.index:

                value = safe_value(cow, col)

                details[col.replace("_", " ").title()] = value

        if details:

            detail_df = pd.DataFrame(
                list(details.items()),
                columns=["Parameter", "Value"],
            )

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # FEATURE CONTRIBUTIONS
    # --------------------------------------------------------

    st.markdown("### 🧠 AI Risk Drivers")

    contributions = None

    try:

        if hasattr(MODEL, "get_booster"):

            row = cow[FEATURES].to_frame().T

            booster = MODEL.get_booster()

            import xgboost as xgb

            matrix = xgb.DMatrix(
                row,
                feature_names=FEATURES
            )

            values = booster.predict(
                matrix,
                pred_contribs=True
            )[0]

            values = values[:-1]

            contributions = pd.Series(
                values,
                index=FEATURES
            )

    except Exception:

        contributions = None

    if contributions is not None:

        contribution_df = (
            contributions
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
            title="Top AI Risk Contributors",
        )

        fig.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=50, b=10),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "AI feature contribution is unavailable "
            "for this saved model."
        )


# ============================================================
# TAB 3 — WHAT IF SIMULATOR
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">🧪 What-If Risk Simulator</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Adjust selected cow parameters to explore how "
        "risk may change. This is a simulation, not a diagnosis."
    )

    simulator_cow = st.selectbox(
        "Select a reference cow",
        cow_options,
        key="simulator_cow",
    )

    if id_column is None:

        base_cow = DF.loc[simulator_cow].copy()

    else:

        base_cow = DF[
            DF[id_column] == simulator_cow
        ].iloc[0].copy()

    simulation = base_cow[FEATURES].copy()

    # --------------------------------------------------------
    # SELECT NUMERIC FEATURES
    # --------------------------------------------------------

    numeric_features = []

    for feature in FEATURES:

        try:

            value = pd.to_numeric(
                base_cow[feature]
            )

            if pd.notna(value):

                numeric_features.append(feature)

        except Exception:
            pass

    editable_features = [
        feature
        for feature in [
            "scc",
            "conductivity",
            "yield_l",
            "body_temp",
            "humidity",
            "thi",
            "hygiene",
        ]
        if feature in numeric_features
    ]

    if not editable_features:

        editable_features = numeric_features[:7]

    st.markdown("### Modify Parameters")

    cols = st.columns(2)

    for i, feature in enumerate(editable_features):

        current_value = float(
            pd.to_numeric(
                base_cow[feature]
            )
        )

        minimum = current_value * 0.50

        maximum = current_value * 1.50

        if current_value == 0:

            minimum = -1
            maximum = 1

        with cols[i % 2]:

            new_value = st.number_input(
                feature.replace("_", " ").title(),
                value=current_value,
                min_value=float(minimum),
                max_value=float(maximum),
                step=max(
                    abs(current_value) * 0.01,
                    0.01
                ),
            )

            simulation[feature] = new_value

    # --------------------------------------------------------
    # SIMULATE
    # --------------------------------------------------------

    try:

        simulated_probability = float(
            MODEL.predict_proba(
                pd.DataFrame(
                    [simulation],
                    columns=FEATURES
                )
            )[0][1]
        )

        original_probability = float(
            MODEL.predict_proba(
                pd.DataFrame(
                    [base_cow[FEATURES]],
                    columns=FEATURES
                )
            )[0][1]
        )

        change = (
            simulated_probability
            - original_probability
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Original Risk",
                f"{original_probability * 100:.1f}%"
            )

        with c2:

            st.metric(
                "Simulated Risk",
                f"{simulated_probability * 100:.1f}%"
            )

        with c3:

            st.metric(
                "Risk Change",
                f"{change * 100:+.1f}%"
            )

        simulated_band = risk_band(
            simulated_probability,
            high_threshold,
            medium_threshold
        )

        st.markdown(
            f"""
            <div class="info-box">
                <b>Simulation Result:</b>
                The predicted risk level is
                <b>{simulated_band}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:

        st.error(
            f"Simulation failed: {e}"
        )


# ============================================================
# TAB 4 — AI PERFORMANCE
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">📈 AI Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Performance metrics are calculated only when "
        "appropriate ground-truth labels are available in the dataset."
    )

    # --------------------------------------------------------
    # FIND LABEL COLUMN
    # --------------------------------------------------------

    label_column = None

    possible_labels = [
        "label",
        "target",
        "mastitis",
        "mastitis_label",
        "diagnosis",
        "y",
    ]

    for col in possible_labels:

        if col in DF.columns:

            label_column = col
            break

    if label_column is not None:

        y_true = DF[label_column].copy()

        # Convert common string labels
        if y_true.dtype == "object":

            mapping = {
                "yes": 1,
                "no": 0,
                "positive": 1,
                "negative": 0,
                "mastitis": 1,
                "healthy": 0,
                "infected": 1,
                "normal": 0,
            }

            y_true = (
                y_true
                .astype(str)
                .str.lower()
                .map(mapping)
            )

        y_true = pd.to_numeric(
            y_true,
            errors="coerce"
        )

        valid = y_true.notna()

        y_true = y_true[valid].astype(int)

        y_prob = DF.loc[
            valid,
            "risk"
        ]

        y_pred = (
            y_prob >= 0.50
        ).astype(int)

        if len(y_true) > 0 and y_true.nunique() == 2:

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            auc = roc_auc_score(
                y_true,
                y_prob
            )

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

            # ------------------------------------------------
            # ROC CURVE
            # ------------------------------------------------

            fpr, tpr, _ = roc_curve(
                y_true,
                y_prob
            )

            roc_df = pd.DataFrame(
                {
                    "False Positive Rate": fpr,
                    "True Positive Rate": tpr,
                }
            )

            fig = px.line(
                roc_df,
                x="False Positive Rate",
                y="True Positive Rate",
                title=f"ROC Curve — AUC {auc:.3f}",
            )

            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    dash="dash"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ------------------------------------------------
            # PRECISION RECALL
            # ------------------------------------------------

            precision_curve, recall_curve, _ = (
                precision_recall_curve(
                    y_true,
                    y_prob
                )
            )

            pr_df = pd.DataFrame(
                {
                    "Recall": recall_curve,
                    "Precision": precision_curve,
                }
            )

            fig = px.line(
                pr_df,
                x="Recall",
                y="Precision",
                title="Precision–Recall Curve",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            cm = confusion_matrix(
                y_true,
                y_pred
            )

            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual Healthy",
                    "Actual Mastitis",
                ],
                columns=[
                    "Predicted Healthy",
                    "Predicted Mastitis",
                ],
            )

            fig = px.imshow(
                cm_df,
                text_auto=True,
                title="Confusion Matrix",
                labels={
                    "x": "Prediction",
                    "y": "Actual",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "Ground-truth labels are not suitable "
                "for calculating binary classification metrics."
            )

    else:

        st.warning(
            "No ground-truth label column was found in "
            "`mastitis_features.csv`."
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown("### 🔍 Feature Importance")

    try:

        if hasattr(MODEL, "feature_importances_"):

            importance = pd.DataFrame(
                {
                    "Feature": FEATURES,
                    "Importance": MODEL.feature_importances_,
                }
            ).sort_values(
                "Importance",
                ascending=False
            ).head(15)

            fig = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Model Features",
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
                "for this model type."
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
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">

        <h3>🐄 What is MastiWatch?</h3>

        MastiWatch is an AI-based early warning platform
        designed to identify cows that may be at increased
        risk of bovine mastitis.

        Instead of waiting until visible clinical symptoms
        appear, the system analyzes available cow-level
        and environmental indicators to estimate risk.

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🤖 AI Pipeline")

        st.markdown(
            """
            **1. Data Collection**

            Cow health, milk quality,
            environmental and management parameters.

            **2. Feature Engineering**

            Moving averages, changes,
            deviations and derived indicators.

            **3. Machine Learning**

            XGBoost-based classification model.

            **4. Risk Prediction**

            Each cow receives a probability score.

            **5. Early Warning**

            Cows are grouped into Low,
            Medium and High risk categories.
            """
        )

    with col2:

        st.markdown("### 🚨 Recommended Workflow")

        st.markdown(
            """
            **Low Risk**

            Continue routine monitoring.

            **Medium Risk**

            Increase observation and monitoring.

            **High Risk**

            Prioritize the cow for veterinary
            inspection and appropriate follow-up.

            **Important**

            MastiWatch provides an AI-based
            risk estimate and does not replace
            professional veterinary diagnosis.
            """
        )

    st.divider()

    st.markdown("### 👩‍💻 Team — Med Sphere")

    team = pd.DataFrame(
        {
            "Team Member": [
                "Asmiyanaseem S",
                "Rakshita M",
                "Sherifa Beevi N",
                "Srirammuthiah C",
            ],
            "Institution": [
                "PSNA College of Engineering and Technology",
                "PSNA College of Engineering and Technology",
                "PSNA College of Engineering and Technology",
                "PSNA College of Engineering and Technology",
            ],
        }
    )

    st.dataframe(
        team,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown(
        """
        <div style="text-align:center;padding:20px;color:#6b7280;">

        <b>MastiWatch</b><br>
        AI-Based Bovine Mastitis Early Warning System<br><br>

        Built for innovation, preventive monitoring
        and smarter livestock healthcare.

        <br><br>

        ⚠️ Research Prototype — Not a Medical/Veterinary Diagnostic Tool

        </div>
        """,
        unsafe_allow_html=True,
    )
