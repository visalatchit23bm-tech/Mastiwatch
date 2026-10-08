# =====================================================================
# MASTIWATCH
# AI-BASED EARLY FORECASTING OF BOVINE MASTITIS
# Team Med Sphere | PSNA College of Engineering and Technology
#
# Required files:
#   app.py
#   mastitis_model.pkl
#   mastitis_features.csv
#
# Run:
#   streamlit run app.py
# =====================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------

BASE = Path(__file__).parent

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="MastiWatch | AI Mastitis Early Warning",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* =========================================================
       GLOBAL
       ========================================================= */

    .stApp {
        background:
            linear-gradient(
                180deg,
                #f7fbf8 0%,
                #ffffff 40%,
                #f7faf8 100%
            );
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        font-family: "Inter", "Segoe UI", sans-serif;
        color: #173b2b;
    }

    p, div, span, label {
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    /* =========================================================
       SIDEBAR
       ========================================================= */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0f5132 0%,
                #164f36 50%,
                #0b3d27 100%
            );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stSelectbox div,
    section[data-testid="stSidebar"] .stSlider div {
        color: #173b2b !important;
    }

    /* =========================================================
       BRAND HEADER
       ========================================================= */

    .brand-header {
        background:
            linear-gradient(
                135deg,
                #0f5132 0%,
                #18794e 55%,
                #239b66 100%
            );

        padding: 28px 32px;
        border-radius: 20px;
        color: white;

        box-shadow:
            0 10px 30px rgba(15, 81, 50, 0.18);

        margin-bottom: 25px;
    }

    .brand-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .brand-subtitle {
        font-size: 15px;
        opacity: 0.92;
        margin-top: 5px;
    }

    .brand-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 12px;
        margin-top: 14px;
    }

    /* =========================================================
       DASHBOARD CARDS
       ========================================================= */

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 16px;

        border: 1px solid #e5eee8;

        box-shadow:
            0 5px 18px rgba(18, 66, 42, 0.06);

        min-height: 125px;
    }

    .metric-label {
        font-size: 13px;
        color: #718078;
        font-weight: 600;
        margin-bottom: 7px;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 800;
        color: #173b2b;
    }

    .metric-small {
        font-size: 12px;
        color: #829088;
        margin-top: 4px;
    }

    .metric-high {
        border-left: 5px solid #dc3545;
    }

    .metric-medium {
        border-left: 5px solid #f39c12;
    }

    .metric-low {
        border-left: 5px solid #28a745;
    }

    .metric-total {
        border-left: 5px solid #18794e;
    }

    /* =========================================================
       SECTION HEADERS
       ========================================================= */

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #173b2b;
        margin-top: 15px;
        margin-bottom: 4px;
    }

    .section-description {
        color: #718078;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* =========================================================
       RISK BADGES
       ========================================================= */

    .risk-high {
        display: inline-block;
        background: #fff0f1;
        color: #c62828;
        border: 1px solid #ffc7cb;
        padding: 7px 15px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 13px;
    }

    .risk-medium {
        display: inline-block;
        background: #fff8e8;
        color: #b86b00;
        border: 1px solid #ffe0a3;
        padding: 7px 15px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 13px;
    }

    .risk-low {
        display: inline-block;
        background: #edf9f0;
        color: #218838;
        border: 1px solid #bde7c7;
        padding: 7px 15px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 13px;
    }

    /* =========================================================
       COW PROFILE
       ========================================================= */

    .cow-profile {
        background: white;
        border: 1px solid #e5eee8;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 5px 18px rgba(18, 66, 42, 0.05);
    }

    .cow-id {
        font-size: 25px;
        font-weight: 800;
        color: #173b2b;
    }

    .profile-item {
        margin-top: 9px;
        color: #5d6c63;
        font-size: 14px;
    }

    .profile-item strong {
        color: #173b2b;
    }

    /* =========================================================
       INFO CARDS
       ========================================================= */

    .info-box {
        background: #f2f8f4;
        border: 1px solid #d8eade;
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 10px;
        color: #345344;
    }

    .warning-box {
        background: #fff8e9;
        border: 1px solid #f5dda6;
        border-radius: 14px;
        padding: 16px 18px;
        color: #745317;
    }

    .danger-box {
        background: #fff1f2;
        border: 1px solid #f4c5c9;
        border-radius: 14px;
        padding: 16px 18px;
        color: #8d2831;
    }

    /* =========================================================
       FOOTER
       ========================================================= */

    .footer {
        margin-top: 45px;
        padding: 25px;
        text-align: center;

        border-top: 1px solid #e1ebe5;

        color: #75837b;
        font-size: 12px;
    }

    /* =========================================================
       TABS
       ========================================================= */

    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    /* =========================================================
       DATAFRAME
       ========================================================= */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------------

RED = "#dc3545"
ORANGE = "#f39c12"
GREEN = "#28a745"
BLUE = "#18794e"
DARK_GREEN = "#0f5132"

BAND_COLOUR = {
    "HIGH": RED,
    "MEDIUM": ORANGE,
    "LOW": GREEN,
}

BAND_ICON = {
    "HIGH": "🔴",
    "MEDIUM": "🟠",
    "LOW": "🟢",
}

# ---------------------------------------------------------------------
# TEXT
# ---------------------------------------------------------------------

TEXT = {

    "English": {

        "title": "MastiWatch",
        "subtitle":
            "AI-powered early warning system for bovine mastitis",

        "description":
            "Forecast mastitis risk before visible symptoms appear "
            "using routinely recorded farm data.",

        "language": "Language / மொழி",

        "settings": "Dashboard Settings",

        "day": "Monitoring Day",
        "breed": "Breed Filter",
        "all": "All Breeds",

        "thresholds": "Risk Thresholds",
        "high_threshold": "High risk from",
        "medium_threshold": "Medium risk from",

        "tabs": [
            "🏠 Herd Dashboard",
            "🐄 Cow Intelligence",
            "🧪 What-If Simulator",
            "📊 AI Performance",
            "ℹ️ About"
        ],

        "herd_title": "Herd Risk Intelligence",
        "herd_description":
            "Real-time overview of mastitis risk across the monitored herd.",

        "total": "Total Cows",
        "high": "High Risk",
        "medium": "Medium Risk",
        "low": "Low Risk",

        "distribution": "Risk Distribution",
        "breed_risk": "Average Risk by Breed",

        "risk_table": "Priority Cow List",

        "rows": "Cows to display",

        "download": "⬇️ Download Alert Report",

        "select_cow": "Select Cow",

        "risk_title": "Mastitis Risk — Next 5 Days",

        "why": "Why is this cow flagged?",

        "why_description":
            "Feature contributions show which measurements increase "
            "or decrease the predicted risk.",

        "trend": "Cow Health Trends",

        "risk_trend": "Risk Trend",

        "profile": "Cow Profile",

        "what_if": "What-If Risk Simulator",

        "what_if_description":
            "Change selected health indicators and instantly observe "
            "how predicted risk responds.",

        "original": "Original Risk",
        "simulated": "Simulated Risk",

        "performance":
            "Model Performance",

        "about": "About MastiWatch",

        "no_data":
            "No measurement is available for this cow on the selected day.",

        "actions": {

            "HIGH":
                "Immediate veterinary attention is recommended. "
                "Perform CMT testing, review milking hygiene, "
                "and follow farm veterinary protocols.",

            "MEDIUM":
                "Monitor this cow closely. Repeat relevant milk "
                "measurements and maintain strict udder hygiene.",

            "LOW":
                "Continue routine monitoring and standard milking hygiene.",

        },

        "demo_note":
            "⚠️ Research prototype: current model results are based "
            "on simulated/demo data and require validation on real "
            "farm records before clinical or field deployment.",
    },

    "தமிழ்": {

        "title": "MastiWatch",

        "subtitle":
            "மாடுகளில் மடிவீக்க நோயை முன்கூட்டியே கண்டறியும் AI அமைப்பு",

        "description":
            "வெளிப்படையான அறிகுறிகள் தோன்றுவதற்கு முன்பே "
            "மடிவீக்க நோய் ஆபத்தை கணிக்கிறது.",

        "language": "Language / மொழி",

        "settings": "அமைப்புகள்",

        "day": "கண்காணிப்பு நாள்",
        "breed": "இன வடிகட்டி",
        "all": "அனைத்து இனங்கள்",

        "thresholds": "ஆபத்து எல்லைகள்",
        "high_threshold": "அதிக ஆபத்து தொடக்கம்",
        "medium_threshold": "நடுத்தர ஆபத்து தொடக்கம்",

        "tabs": [
            "🏠 மந்தை மேலோட்டம்",
            "🐄 மாடு விவரம்",
            "🧪 என்ன நடந்தால்?",
            "📊 AI செயல்திறன்",
            "ℹ️ MastiWatch பற்றி"
        ],

        "herd_title": "மந்தை ஆபத்து நிலை",

        "herd_description":
            "மந்தையில் உள்ள மாடுகளின் மடிவீக்க ஆபத்தை ஒரே இடத்தில் கண்காணிக்கவும்.",

        "total": "மொத்த மாடுகள்",
        "high": "அதிக ஆபத்து",
        "medium": "நடுத்தர ஆபத்து",
        "low": "குறைந்த ஆபத்து",

        "distribution": "ஆபத்து பகிர்வு",
        "breed_risk": "இனம் வாரியான சராசரி ஆபத்து",

        "risk_table": "முக்கிய கவனம் தேவைப்படும் மாடுகள்",

        "rows": "காட்ட வேண்டிய மாடுகள்",

        "download": "⬇️ எச்சரிக்கை அறிக்கையை பதிவிறக்கவும்",

        "select_cow": "மாட்டை தேர்ந்தெடுக்கவும்",

        "risk_title": "அடுத்த 5 நாட்களுக்கான மடிவீக்க ஆபத்து",

        "why": "இந்த மாடு ஏன் எச்சரிக்கப்பட்டது?",

        "why_description":
            "எந்த அளவீடுகள் ஆபத்தை அதிகரிக்கின்றன அல்லது குறைக்கின்றன என்பதை பார்க்கலாம்.",

        "trend": "மாட்டின் உடல்நிலை போக்குகள்",

        "risk_trend": "ஆபத்து போக்கு",

        "profile": "மாட்டின் விவரம்",

        "what_if": "What-If ஆபத்து சிமுலேட்டர்",

        "what_if_description":
            "மாட்டின் அளவீடுகளை மாற்றி ஆபத்து எப்படி மாறுகிறது என்பதைப் பாருங்கள்.",

        "original": "அசல் ஆபத்து",
        "simulated": "சிமுலேட் ஆபத்து",

        "performance": "மாதிரி செயல்திறன்",

        "about": "MastiWatch பற்றி",

        "no_data":
            "தேர்ந்தெடுத்த நாளில் இந்த மாட்டிற்கான தரவு இல்லை.",

        "actions": {

            "HIGH":
                "உடனடி கால்நடை மருத்துவ பரிசோதனை பரிந்துரைக்கப்படுகிறது. "
                "CMT பரிசோதனை மற்றும் சுகாதார நிலையை சரிபார்க்கவும்.",

            "MEDIUM":
                "இந்த மாட்டை கவனமாக கண்காணிக்கவும். "
                "தேவையான பால் அளவீடுகளை மீண்டும் பரிசோதிக்கவும்.",

            "LOW":
                "வழக்கமான கண்காணிப்பு மற்றும் சுத்தமான பால் கறக்கும் முறையை தொடரவும்.",

        },

        "demo_note":
            "⚠️ இது ஒரு ஆராய்ச்சி prototype. தற்போதைய முடிவுகள் simulated/demo "
            "data அடிப்படையிலானவை; உண்மையான farm data மூலம் validation தேவை.",
    }
}

# ---------------------------------------------------------------------
# FEATURE NAMES
# ---------------------------------------------------------------------

NICE = {

    "scc_ma3": "SCC — 3 Day Average",
    "scc_ma7": "SCC — 7 Day Average",
    "scc_log": "SCC — Log",
    "scc_dev": "SCC vs Cow Normal",
    "scc_chg3": "SCC Change",

    "conductivity": "Milk Conductivity",
    "conductivity_ma3": "Conductivity — 3 Day Avg",
    "conductivity_ma7": "Conductivity — 7 Day Avg",
    "conductivity_dev": "Conductivity vs Normal",
    "conductivity_chg3": "Conductivity Change",

    "yield_l": "Milk Yield",
    "yield_l_ma3": "Milk Yield — 3 Day Avg",
    "yield_l_ma7": "Milk Yield — 7 Day Avg",
    "yield_l_dev": "Yield vs Normal",
    "yield_l_chg3": "Yield Change",

    "body_temp": "Body Temperature",
    "body_temp_ma3": "Body Temperature — 3 Day Avg",
    "body_temp_ma7": "Body Temperature — 7 Day Avg",
    "body_temp_dev": "Body Temperature vs Normal",
    "body_temp_chg3": "Body Temperature Change",

    "hand_milking": "Hand Milking",
    "hygiene": "Hygiene Score",
    "parity": "Number of Calvings",
    "humidity": "Humidity",
    "thi": "Heat Stress — THI",
    "thi_ma3": "Heat Stress — 3 Day Avg",
    "temp": "Air Temperature",
    "dim": "Days in Milk",
    "breed_code": "Breed",
}

# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------

@st.cache_resource
def load_model():

    bundle = joblib.load(
        BASE / "mastitis_model.pkl"
    )

    return bundle["model"], bundle["features"]


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

@st.cache_data
def load_scored_data():

    model, features = load_model()

    df = pd.read_csv(
        BASE / "mastitis_features.csv"
    )

    df["risk"] = model.predict_proba(
        df[features]
    )[:, 1]

    return df


# ---------------------------------------------------------------------
# RISK BAND
# ---------------------------------------------------------------------

def get_band(probability, high, medium):

    if probability >= high:
        return "HIGH"

    if probability >= medium:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------------------
# XGBOOST CONTRIBUTIONS
# ---------------------------------------------------------------------

def get_contributions(model, features, row):

    try:

        import xgboost as xgb

        matrix = xgb.DMatrix(
            row[features]
        )

        values = model.get_booster().predict(
            matrix,
            pred_contribs=True
        )[0][:-1]

        return pd.Series(
            values,
            index=features
        )

    except Exception:

        return None


# ---------------------------------------------------------------------
# GAUGE
# ---------------------------------------------------------------------

def create_gauge(probability, high, medium):

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,

            number={
                "suffix": "%",
                "font": {
                    "size": 38,
                    "color": DARK_GREEN
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1
                },

                "bar": {
                    "color": DARK_GREEN,
                    "thickness": 0.25
                },

                "steps": [

                    {
                        "range": [
                            0,
                            medium * 100
                        ],
                        "color": "#e7f6ea"
                    },

                    {
                        "range": [
                            medium * 100,
                            high * 100
                        ],
                        "color": "#fff1d7"
                    },

                    {
                        "range": [
                            high * 100,
                            100
                        ],
                        "color": "#ffe4e7"
                    }
                ]
            }
        )
    )

    figure.update_layout(
        height=250,
        margin=dict(
            l=10,
            r=10,
            t=15,
            b=5
        )
    )

    return figure


# ---------------------------------------------------------------------
# LINE CHART
# ---------------------------------------------------------------------

def create_line_chart(
    dataframe,
    column,
    title,
    color,
    selected_day=None
):

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=dataframe["day"],
            y=dataframe[column],
            mode="lines+markers",

            line={
                "color": color,
                "width": 3
            },

            marker={
                "size": 6
            }
        )
    )

    if selected_day is not None:

        figure.add_vline(
            x=selected_day,
            line_dash="dot",
            line_color="#888888"
        )

    figure.update_layout(

        title={
            "text": title,
            "font": {
                "size": 15
            }
        },

        height=300,

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),

        paper_bgcolor="white",
        plot_bgcolor="white",

        xaxis={
            "title": "Day",
            "gridcolor": "#edf2ef"
        },

        yaxis={
            "gridcolor": "#edf2ef"
        },

        hovermode="x unified"
    )

    return figure


# ---------------------------------------------------------------------
# LEAD TIME
# ---------------------------------------------------------------------

def calculate_lead_time(df, threshold):

    results = []

    if "onset_day" not in df.columns:
        return float("nan")

    for _, group in df[
        df["onset_day"] >= 0
    ].groupby("cow_id"):

        onset = group["onset_day"].iloc[0]

        before = group[
            (group["day"] < onset)
            &
            (group["day"] >= onset - 5)
            &
            (group["risk"] >= threshold)
        ]

        if len(before):

            results.append(
                onset - before["day"].min()
            )

    if not results:
        return float("nan")

    return float(
        np.mean(results)
    )


# =====================================================================
# SIDEBAR
# =====================================================================

lang = st.sidebar.radio(
    "🌐 Language / மொழி",
    ["English", "தமிழ்"]
)

T = TEXT[lang]

st.sidebar.markdown(
    """
    <div style="
        padding:15px;
        border-radius:14px;
        background:rgba(255,255,255,0.10);
        margin-bottom:20px;
        text-align:center;
    ">
        <div style="font-size:36px;">🐄</div>
        <div style="font-size:20px;font-weight:800;">
            MastiWatch
        </div>
        <div style="font-size:11px;opacity:0.8;">
            AI Mastitis Early Warning
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

model, FEATURES = load_model()

DF = load_scored_data()

st.sidebar.header(
    T["settings"]
)

# Day

days = sorted(
    DF["day"].unique()
)

selected_day = st.sidebar.select_slider(
    T["day"],
    options=days,
    value=days[-1]
)

# Breed

breeds = [
    T["all"]
] + sorted(
    DF["breed"].unique()
)

selected_breed = st.sidebar.selectbox(
    T["breed"],
    breeds
)

# Thresholds

st.sidebar.markdown(
    f"**{T['thresholds']}**"
)

HIGH = st.sidebar.slider(
    T["high_threshold"],
    0.30,
    0.95,
    0.60,
    0.05
)

MEDIUM = st.sidebar.slider(
    T["medium_threshold"],
    0.05,
    float(HIGH) - 0.05,
    min(0.30, HIGH - 0.05),
    0.05
)

# Add risk band

DF["band"] = DF["risk"].apply(
    lambda x: get_band(
        x,
        HIGH,
        MEDIUM
    )
)

# Filter breed

if selected_breed == T["all"]:

    VIEW = DF.copy()

else:

    VIEW = DF[
        DF["breed"] == selected_breed
    ].copy()

# =====================================================================
# HEADER
# =====================================================================

st.markdown(
    f"""
    <div class="brand-header">

        <div class="brand-title">
            🐄 {T["title"]}
        </div>

        <div class="brand-subtitle">
            {T["subtitle"]}
        </div>

        <div class="brand-subtitle">
            {T["description"]}
        </div>

        <div class="brand-badge">
            🤖 XGBoost &nbsp; • &nbsp;
            🧠 Explainable AI &nbsp; • &nbsp;
            📅 5-Day Forecast
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================================
# CURRENT HERD DATA
# =====================================================================

latest = (
    VIEW[
        VIEW["day"] <= selected_day
    ]
    .sort_values("day")
    .groupby("cow_id")
    .tail(1)
)

latest = latest[
    latest["day"] == selected_day
]

today = latest.sort_values(
    "risk",
    ascending=False
)

# =====================================================================
# NAVIGATION
# =====================================================================

tabs = st.tabs(
    T["tabs"]
)

tab_herd = tabs[0]
tab_cow = tabs[1]
tab_whatif = tabs[2]
tab_perf = tabs[3]
tab_about = tabs[4]

# =====================================================================
# TAB 1 — HERD DASHBOARD
# =====================================================================

with tab_herd:

    st.markdown(
        f"""
        <div class="section-title">
            {T["herd_title"]}
        </div>

        <div class="section-description">
            {T["herd_description"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    total_cows = len(today)

    high_count = int(
        (today["band"] == "HIGH").sum()
    )

    medium_count = int(
        (today["band"] == "MEDIUM").sum()
    )

    low_count = int(
        (today["band"] == "LOW").sum()
    )

    # -------------------------------------------------------------
    # METRIC CARDS
    # -------------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card metric-total">

                <div class="metric-label">
                    🐄 {T["total"]}
                </div>

                <div class="metric-value">
                    {total_cows}
                </div>

                <div class="metric-small">
                    Monitored today
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card metric-high">

                <div class="metric-label">
                    🔴 {T["high"]}
                </div>

                <div class="metric-value">
                    {high_count}
                </div>

                <div class="metric-small">
                    Immediate attention
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card metric-medium">

                <div class="metric-label">
                    🟠 {T["medium"]}
                </div>

                <div class="metric-value">
                    {medium_count}
                </div>

                <div class="metric-small">
                    Close monitoring
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card metric-low">

                <div class="metric-label">
                    🟢 {T["low"]}
                </div>

                <div class="metric-value">
                    {low_count}
                </div>

                <div class="metric-small">
                    Routine monitoring
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if today.empty:

        st.warning(
            T["no_data"]
        )

    else:

        # ---------------------------------------------------------
        # CHARTS
        # ---------------------------------------------------------

        chart1, chart2 = st.columns(2)

        with chart1:

            counts = (
                today["band"]
                .value_counts()
                .reindex(
                    ["HIGH", "MEDIUM", "LOW"]
                )
                .fillna(0)
            )

            pie = go.Figure(
                go.Pie(

                    labels=counts.index,

                    values=counts.values,

                    hole=0.60,

                    marker=dict(
                        colors=[
                            RED,
                            ORANGE,
                            GREEN
                        ]
                    ),

                    textinfo="label+percent",

                    hovertemplate=
                    "<b>%{label}</b><br>"
                    "Cows: %{value}<br>"
                    "%{percent}"
                    "<extra></extra>"
                )
            )

            pie.update_layout(

                title=T["distribution"],

                height=330,

                paper_bgcolor="white",

                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10
                ),

                legend=dict(
                    orientation="h",
                    y=-0.05
                )
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )

        with chart2:

            breed_risk = (
                today
                .groupby("breed")["risk"]
                .mean()
                .mul(100)
                .sort_values()
            )

            bar = go.Figure(
                go.Bar(

                    x=breed_risk.values,

                    y=breed_risk.index,

                    orientation="h",

                    marker=dict(
                        color=BLUE,
                        line=dict(
                            width=0
                        )
                    ),

                    hovertemplate=
                    "<b>%{y}</b><br>"
                    "Average risk: %{x:.1f}%"
                    "<extra></extra>"
                )
            )

            bar.update_layout(

                title=T["breed_risk"],

                height=330,

                paper_bgcolor="white",

                plot_bgcolor="white",

                xaxis=dict(
                    title="Risk (%)",
                    gridcolor="#edf2ef"
                ),

                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10
                )
            )

            st.plotly_chart(
                bar,
                use_container_width=True
            )

        # ---------------------------------------------------------
        # ALERT TABLE
        # ---------------------------------------------------------

        st.markdown(
            f"""
            <div class="section-title">
                🚨 {T["risk_table"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        maximum = min(
            50,
            max(5, len(today))
        )

        default_rows = min(
            15,
            len(today)
        )

        number_rows = st.slider(
            T["rows"],
            5,
            maximum,
            default_rows
        )

        table = today[
            [
                "cow_id",
                "breed",
                "risk",
                "band",
                "yield_l",
                "conductivity",
                "scc",
                "body_temp",
            ]
        ].head(number_rows).copy()

        table["Risk"] = (
            table["risk"] * 100
        ).round(1).astype(str) + "%"

        table["Risk Level"] = (
            table["band"]
            .map(
                lambda x:
                f"{BAND_ICON[x]} {x}"
            )
        )

        table = table[
            [
                "cow_id",
                "breed",
                "Risk",
                "Risk Level",
                "yield_l",
                "conductivity",
                "scc",
                "body_temp",
            ]
        ]

        table.columns = [
            "Cow ID",
            "Breed",
            "Risk",
            "Risk Level",
            "Milk Yield (L)",
            "Conductivity",
            "SCC",
            "Body Temp (°C)"
        ]

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=430
        )

        st.download_button(
            label=T["download"],
            data=today.to_csv(
                index=False
            ).encode("utf-8"),

            file_name=
            f"MastiWatch_Alerts_Day_{selected_day}.csv",

            mime="text/csv"
        )

# =====================================================================
# TAB 2 — COW INTELLIGENCE
# =====================================================================

with tab_cow:

    st.markdown(
        f"""
        <div class="section-title">
            🐄 {T["tabs"][1]}
        </div>

        <div class="section-description">
            Individual cow risk profile, trends and explainable AI insights.
        </div>
        """,
        unsafe_allow_html=True
    )

    cow_ids = sorted(
        VIEW["cow_id"].unique()
    )

    if not cow_ids:

        st.warning(
            T["no_data"]
        )

    else:

        default_cow = (
            int(today.iloc[0]["cow_id"])
            if len(today)
            else cow_ids[0]
        )

        selected_cow = st.selectbox(
            T["select_cow"],
            cow_ids,
            index=cow_ids.index(
                default_cow
            )
        )

        cow_df = (
            DF[
                DF["cow_id"] == selected_cow
            ]
            .sort_values("day")
        )

        current = (
            cow_df[
                cow_df["day"] <= selected_day
            ]
            .tail(1)
        )

        if current.empty:

            st.warning(
                T["no_data"]
            )

        else:

            row = current.iloc[0]

            probability = float(
                row["risk"]
            )

            risk_band = get_band(
                probability,
                HIGH,
                MEDIUM
            )

            left, right = st.columns(
                [1, 1.7]
            )

            # -----------------------------------------------------
            # LEFT
            # -----------------------------------------------------

            with left:

                st.markdown(
                    f"""
                    <div class="cow-profile">

                        <div class="cow-id">
                            🐄 Cow {selected_cow}
                        </div>

                        <div class="profile-item">
                            <strong>Breed:</strong>
                            {row["breed"]}
                        </div>

                        <div class="profile-item">
                            <strong>Calvings:</strong>
                            {int(row["parity"])}
                        </div>

                        <div class="profile-item">
                            <strong>Days in milk:</strong>
                            {int(row["dim"])}
                        </div>

                        <div class="profile-item">
                            <strong>Hygiene score:</strong>
                            {int(row["hygiene"])}/5
                        </div>

                        <div class="profile-item">
                            <strong>Hand milking:</strong>
                            {"Yes" if row["hand_milking"] else "No"}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"### {T['risk_title']}"
                )

                st.plotly_chart(
                    create_gauge(
                        probability,
                        HIGH,
                        MEDIUM
                    ),
                    use_container_width=True
                )

                badge_class = {
                    "HIGH": "risk-high",
                    "MEDIUM": "risk-medium",
                    "LOW": "risk-low"
                }[risk_band]

                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <span class="{badge_class}">
                            {BAND_ICON[risk_band]}
                            {risk_band}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if risk_band == "HIGH":

                    st.markdown(
                        f"""
                        <div class="danger-box">
                            <strong>⚠ Immediate Attention</strong><br><br>
                            {T["actions"][risk_band]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif risk_band == "MEDIUM":

                    st.markdown(
                        f"""
                        <div class="warning-box">
                            <strong>⚠ Monitor Closely</strong><br><br>
                            {T["actions"][risk_band]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="info-box">
                            <strong>✓ Routine Monitoring</strong><br><br>
                            {T["actions"][risk_band]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # -----------------------------------------------------
            # RIGHT — EXPLAINABLE AI
            # -----------------------------------------------------

            with right:

                st.markdown(
                    f"""
                    <div class="section-title">
                        🧠 {T["why"]}
                    </div>

                    <div class="section-description">
                        {T["why_description"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                contributions = get_contributions(
                    model,
                    FEATURES,
                    current
                )

                if contributions is not None:

                    top_features = (
                        contributions
                        .reindex(
                            contributions
                            .abs()
                            .sort_values(
                                ascending=False
                            )
                            .index
                        )
                        .head(8)
                        .sort_values()
                    )

                    feature_names = [
                        NICE.get(
                            feature,
                            feature
                        )
                        for feature in
                        top_features.index
                    ]

                    colors = [
                        RED if value > 0
                        else GREEN
                        for value in
                        top_features.values
                    ]

                    explanation = go.Figure(
                        go.Bar(

                            x=top_features.values,

                            y=feature_names,

                            orientation="h",

                            marker_color=colors,

                            hovertemplate=
                            "<b>%{y}</b><br>"
                            "Contribution: %{x:.3f}"
                            "<extra></extra>"
                        )
                    )

                    explanation.update_layout(

                        height=380,

                        paper_bgcolor="white",

                        plot_bgcolor="white",

                        margin=dict(
                            l=10,
                            r=10,
                            t=15,
                            b=15
                        ),

                        xaxis=dict(
                            title="Contribution to Risk",
                            gridcolor="#edf2ef"
                        )
                    )

                    st.plotly_chart(
                        explanation,
                        use_container_width=True
                    )

                    st.caption(
                        "🔴 Factors increasing risk   "
                        "🟢 Factors reducing risk"
                    )

                else:

                    st.info(
                        "Explainable AI information is "
                        "not available for this model."
                    )

            # -----------------------------------------------------
            # RISK TREND
            # -----------------------------------------------------

            st.markdown(
                f"""
                <div class="section-title">
                    📈 {T["trend"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.plotly_chart(
                create_line_chart(
                    cow_df,
                    "risk",
                    T["risk_trend"],
                    RED,
                    selected_day
                ),
                use_container_width=True
            )

            # -----------------------------------------------------
            # HEALTH TRENDS
            # -----------------------------------------------------

            t1, t2 = st.columns(2)

            with t1:

                st.plotly_chart(
                    create_line_chart(
                        cow_df,
                        "yield_l",
                        "🥛 Milk Yield",
                        BLUE,
                        selected_day
                    ),
                    use_container_width=True
                )

            with t2:

                st.plotly_chart(
                    create_line_chart(
                        cow_df,
                        "conductivity",
                        "⚡ Milk Conductivity",
                        ORANGE,
                        selected_day
                    ),
                    use_container_width=True
                )

            t3, t4 = st.columns(2)

            with t3:

                st.plotly_chart(
                    create_line_chart(
                        cow_df,
                        "scc",
                        "🧪 Somatic Cell Count",
                        RED,
                        selected_day
                    ),
                    use_container_width=True
                )

            with t4:

                st.plotly_chart(
                    create_line_chart(
                        cow_df,
                        "body_temp",
                        "🌡 Body Temperature",
                        GREEN,
                        selected_day
                    ),
                    use_container_width=True
                )

# =====================================================================
# TAB 3 — WHAT IF
# =====================================================================

with tab_whatif:

    st.markdown(
        f"""
        <div class="section-title">
            🧪 {T["what_if"]}
        </div>

        <div class="section-description">
            {T["what_if_description"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    ids = sorted(
        VIEW["cow_id"].unique()
    )

    if not ids:

        st.warning(
            T["no_data"]
        )

    else:

        default_index = (
            ids.index(
                int(today.iloc[0]["cow_id"])
            )
            if len(today)
            and int(today.iloc[0]["cow_id"]) in ids
            else 0
        )

        whatif_cow = st.selectbox(
            "🐄 Starting Cow",
            ids,
            index=default_index,
            key="whatif_cow"
        )

        base_data = (
            DF[
                (DF["cow_id"] == whatif_cow)
                &
                (DF["day"] <= selected_day)
            ]
            .sort_values("day")
            .tail(1)
        )

        if base_data.empty:

            st.warning(
                T["no_data"]
            )

        else:

            base = base_data.iloc[0]

            st.markdown(
                "### 🧬 Adjust Health Indicators"
            )

            s1, s2, s3 = st.columns(3)

            with s1:

                scc_log = st.slider(
                    "SCC (log scale)",
                    8.0,
                    15.0,
                    float(base["scc_log"]),
                    0.1
                )

                scc_dev = st.slider(
                    "SCC vs Cow Normal",
                    -1.0,
                    10.0,
                    float(
                        np.clip(
                            base["scc_dev"],
                            -1,
                            10
                        )
                    ),
                    0.1
                )

            with s2:

                conductivity = st.slider(
                    "Milk Conductivity",
                    3.5,
                    7.5,
                    float(
                        base["conductivity"]
                    ),
                    0.05
                )

                conductivity_dev = st.slider(
                    "Conductivity vs Normal",
                    -0.5,
                    0.8,
                    float(
                        np.clip(
                            base["conductivity_dev"],
                            -0.5,
                            0.8
                        )
                    ),
                    0.01
                )

            with s3:

                yield_dev = st.slider(
                    "Milk Yield vs Normal",
                    -3.0,
                    3.0,
                    float(
                        np.clip(
                            base["yield_l_dev"],
                            -3,
                            3
                        )
                    ),
                    0.1
                )

                body_temperature = st.slider(
                    "Body Temperature",
                    37.5,
                    41.0,
                    float(
                        np.clip(
                            base["body_temp"],
                            37.5,
                            41
                        )
                    ),
                    0.1
                )

            s4, s5 = st.columns(2)

            with s4:

                hygiene = st.slider(
                    "Hygiene Score",
                    1,
                    5,
                    int(base["hygiene"])
                )

            with s5:

                hand_milking = st.radio(
                    "Hand Milking",
                    [0, 1],

                    index=int(
                        base["hand_milking"]
                    ),

                    format_func=lambda x:
                    "Yes" if x else "No",

                    horizontal=True
                )

            # -----------------------------------------------------
            # CREATE SIMULATED INPUT
            # -----------------------------------------------------

            simulated = base_data.copy()

            simulated["scc_log"] = scc_log
            simulated["scc_dev"] = scc_dev

            simulated["conductivity"] = conductivity
            simulated["conductivity_dev"] = conductivity_dev

            simulated["yield_l_dev"] = yield_dev

            simulated["body_temp"] = body_temperature

            simulated["hygiene"] = hygiene

            simulated["hand_milking"] = hand_milking

            new_probability = float(
                model.predict_proba(
                    simulated[FEATURES]
                )[:, 1][0]
            )

            original_probability = float(
                base["risk"]
            )

            # -----------------------------------------------------
            # RESULT
            # -----------------------------------------------------

            r1, r2, r3 = st.columns(3)

            with r1:

                st.metric(
                    T["original"],
                    f"{original_probability * 100:.1f}%"
                )

            with r2:

                delta = (
                    new_probability
                    - original_probability
                ) * 100

                st.metric(
                    T["simulated"],
                    f"{new_probability * 100:.1f}%",
                    f"{delta:+.1f} pts"
                )

            with r3:

                new_band = get_band(
                    new_probability,
                    HIGH,
                    MEDIUM
                )

                st.metric(
                    "Risk Level",
                    f"{BAND_ICON[new_band]} {new_band}"
                )

            if new_band == "HIGH":

                st.error(
                    T["actions"][new_band]
                )

            elif new_band == "MEDIUM":

                st.warning(
                    T["actions"][new_band]
                )

            else:

                st.success(
                    T["actions"][new_band]
                )

# =====================================================================
# TAB 4 — MODEL PERFORMANCE
# =====================================================================

with tab_perf:

    st.markdown(
        f"""
        <div class="section-title">
            📊 {T["performance"]}
        </div>

        <div class="section-description">
            Demonstration of the model's predictive behaviour.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "⚠ The model performance shown here is based on "
        "the project's simulated/demo dataset."
    )

    y_true = DF["label"]

    scores = DF["risk"]

    predictions = (
        scores >= MEDIUM
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    auc = roc_auc_score(
        y_true,
        scores
    )

    lead = calculate_lead_time(
        DF,
        MEDIUM
    )

    # -------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "ROC-AUC",
        f"{auc:.2f}"
    )

    p2.metric(
        "Recall",
        f"{recall:.0%}"
    )

    p3.metric(
        "Precision",
        f"{precision:.2f}"
    )

    p4.metric(
        "Early Warning",
        f"{lead:.1f} days"
        if not np.isnan(lead)
        else "N/A"
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # ROC + PR
    # -------------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true,
            scores
        )

        roc_figure = go.Figure()

        roc_figure.add_trace(
            go.Scatter(
                x=false_positive_rate,
                y=true_positive_rate,
                mode="lines",

                name="MastiWatch",

                line={
                    "color": BLUE,
                    "width": 3
                }
            )
        )

        roc_figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",

                name="Random",

                line={
                    "color": "#999",
                    "dash": "dash"
                }
            )
        )

        roc_figure.update_layout(
            title="ROC Curve",
            height=360,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20
            )
        )

        st.plotly_chart(
            roc_figure,
            use_container_width=True
        )

    with c2:

        precision_values, recall_values, _ = precision_recall_curve(
            y_true,
            scores
        )

        pr_figure = go.Figure()

        pr_figure.add_trace(
            go.Scatter(
                x=recall_values,
                y=precision_values,
                mode="lines",

                line={
                    "color": ORANGE,
                    "width": 3
                },

                name="MastiWatch"
            )
        )

        pr_figure.update_layout(
            title="Precision-Recall Curve",
            height=360,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_title="Recall",
            yaxis_title="Precision",
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20
            )
        )

        st.plotly_chart(
            pr_figure,
            use_container_width=True
        )

    # -------------------------------------------------------------
    # CONFUSION MATRIX + FEATURE IMPORTANCE
    # -------------------------------------------------------------

    c3, c4 = st.columns(2)

    with c3:

        confusion = go.Figure(
            go.Heatmap(

                z=[
                    [tn, fp],
                    [fn, tp]
                ],

                x=[
                    "Predicted Healthy",
                    "Predicted At Risk"
                ],

                y=[
                    "Actual Healthy",
                    "Actual At Risk"
                ],

                text=[
                    [tn, fp],
                    [fn, tp]
                ],

                texttemplate="%{text}",

                colorscale=[
                    [0, "#e7f6ea"],
                    [1, "#18794e"]
                ],

                showscale=False
            )
        )

        confusion.update_layout(
            title="Confusion Matrix",
            height=360,
            paper_bgcolor="white",
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20
            )
        )

        st.plotly_chart(
            confusion,
            use_container_width=True
        )

    with c4:

        importance = (
            pd.Series(
                model.feature_importances_,
                index=FEATURES
            )
            .sort_values()
            .tail(10)
        )

        importance_chart = go.Figure(
            go.Bar(

                x=importance.values,

                y=[
                    NICE.get(
                        x,
                        x
                    )
                    for x in importance.index
                ],

                orientation="h",

                marker_color=BLUE
            )
        )

        importance_chart.update_layout(
            title="Top Model Features",
            height=360,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20
            )
        )

        st.plotly_chart(
            importance_chart,
            use_container_width=True
        )

    st.markdown(
        f"""
        <div class="warning-box">

        <strong>Research Prototype Notice</strong><br><br>

        The MastiWatch prototype uses simulated data.
        The performance metrics should not be interpreted as
        clinical validation. Validation using real farm records
        and veterinary collaboration is required before
        real-world deployment.

        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================================
# TAB 5 — ABOUT
# =====================================================================

with tab_about:

    st.markdown(
        """
        <div class="section-title">
            🐄 About MastiWatch
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        [1.4, 1]
    )

    with col1:

        st.markdown(
            """
            ### What is MastiWatch?

            **MastiWatch** is a software-only AI system designed
            for early forecasting of bovine mastitis.

            Instead of waiting for visible symptoms, the system
            analyses routinely recorded farm information such as:

            - 🥛 Milk yield
            - ⚡ Milk conductivity
            - 🧪 Somatic cell count
            - 🌡 Temperature
            - 💧 Humidity
            - 🐄 Cow characteristics
            - 🧼 Hygiene information
            - 🌡 Heat stress indicators

            The system uses an **XGBoost machine-learning model**
            to estimate mastitis risk and provides an
            explainable AI view of why a cow has been flagged.

            ### AI Pipeline

            **Farm Data → Feature Engineering → XGBoost →
            Risk Score → Explainable AI → Dashboard**

            ### Core Technologies

            **Python · Pandas · Scikit-learn · XGBoost ·
            Plotly · Streamlit**
            """,
        )

    with col2:

        st.markdown(
            """
            ### 🎯 Project Goal

            Detect risk early so farmers and veterinary
            professionals can prioritise cows that need
            attention.

            ### 🌱 Expected Impact

            **Less Milk Loss**

            Earlier intervention may reduce production loss.

            **Healthier Cows**

            Risk-based monitoring can help identify
            potentially affected cows earlier.

            **Smarter Treatment**

            Supports targeted veterinary attention.

            **Scalable**

            Software-based architecture can be adapted
            for farms and cooperatives.

            ### 🇮🇳 Built for India

            The project considers:

            - Heat stress
            - Breed differences
            - Hand milking
            - Hygiene
            - English / Tamil usability
            """
        )

    st.markdown("---")

    st.markdown(
        """
        ### 👥 Team Med Sphere

        **PSNA College of Engineering and Technology**

        Asmiyanaseem S  
        Rakshita M  
        Sherifa Beevi N  
        Srirammuthiah C
        """
    )

    st.markdown(
        f"""
        <div class="warning-box">

        <strong>⚠ Important Limitation</strong><br><br>

        {T["demo_note"]}

        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================================
# FOOTER
# =====================================================================

st.markdown(
    """
    <div class="footer">

        <strong>MastiWatch</strong> · AI-Based Early Mastitis Warning System
        <br><br>

        Team Med Sphere · PSNA College of Engineering and Technology
        <br>

        Detect Early · Act Early · Protect Milk Production

    </div>
    """,
    unsafe_allow_html=True
)
