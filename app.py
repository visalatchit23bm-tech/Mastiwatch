# =====================================================================
# MastiWatch - Mastitis Early-Warning Dashboard
# Team Med Sphere | PSNA College of Engineering and Technology
#
# Files needed in the SAME folder:
#     app.py
#     mastitis_model.pkl
#     mastitis_features.csv
#
# Run:
#     pip install -r requirements.txt
#     streamlit run app.py
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


# =====================================================================
# BASE PATH
# =====================================================================

BASE = Path(__file__).parent


# =====================================================================
# STREAMLIT VERSION COMPATIBILITY
# =====================================================================

import streamlit as _st

try:
    from packaging.version import Version

    STRETCH = (
        {"width": "stretch"}
        if Version(_st.__version__) >= Version("1.50")
        else {"use_container_width": True}
    )

except Exception:
    STRETCH = {"use_container_width": True}


# =====================================================================
# PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="MastiWatch | AI Mastitis Early Warning",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# COLOURS
# =====================================================================

RED = "#d62728"
ORANGE = "#ff9f1c"
GREEN = "#2ca02c"
BLUE = "#1f77b4"

DARK_GREEN = "#14532d"
LIGHT_GREEN = "#dcfce7"
VERY_LIGHT_GREEN = "#f0fdf4"

DARK_TEXT = "#17202a"
GREY = "#64748b"
LIGHT_GREY = "#e2e8f0"

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


# =====================================================================
# PROFESSIONAL UI CSS
# =====================================================================

st.markdown(
    """
    <style>

    /* ---------------------------------------------------------------
       GLOBAL
    --------------------------------------------------------------- */

    .stApp {
        background:
            linear-gradient(
                180deg,
                #f8fcf9 0%,
                #f4faf6 45%,
                #ffffff 100%
            );
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* ---------------------------------------------------------------
       SIDEBAR
    --------------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #f0fdf4 0%,
            #ffffff 100%
        );
        border-right: 1px solid #dbe7df;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #14532d;
    }

    /* ---------------------------------------------------------------
       HERO HEADER
    --------------------------------------------------------------- */

    .mw-hero {
        background:
            linear-gradient(
                135deg,
                #14532d 0%,
                #166534 50%,
                #15803d 100%
            );
        border-radius: 22px;
        padding: 30px 34px;
        color: white;
        margin-bottom: 24px;
        box-shadow:
            0 12px 35px rgba(20, 83, 45, 0.18);
        position: relative;
        overflow: hidden;
    }

    .mw-hero:after {
        content: "🐄";
        position: absolute;
        right: 35px;
        top: 15px;
        font-size: 95px;
        opacity: 0.10;
    }

    .mw-hero-title {
        font-size: 38px;
        font-weight: 850;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .mw-hero-subtitle {
        font-size: 18px;
        margin-top: 7px;
        opacity: 0.95;
        font-weight: 500;
    }

    .mw-hero-description {
        font-size: 14px;
        margin-top: 12px;
        opacity: 0.82;
    }

    .mw-hero-pill {
        display: inline-block;
        margin-top: 16px;
        padding: 6px 13px;
        border-radius: 20px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.20);
        font-size: 12px;
    }

    /* ---------------------------------------------------------------
       SECTION HEADERS
    --------------------------------------------------------------- */

    .mw-section-title {
        color: #14532d;
        font-size: 25px;
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 18px;
    }

    .mw-section-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-top: -12px;
        margin-bottom: 18px;
    }

    /* ---------------------------------------------------------------
       KPI CARDS
    --------------------------------------------------------------- */

    .mw-kpi {
        background: rgba(255,255,255,0.96);
        border: 1px solid #e2e8f0;
        border-radius: 17px;
        padding: 18px 20px;
        min-height: 118px;
        box-shadow: 0 5px 18px rgba(15,23,42,0.055);
        transition: transform 0.2s ease;
    }

    .mw-kpi:hover {
        transform: translateY(-2px);
    }

    .mw-kpi-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 650;
    }

    .mw-kpi-value {
        color: #14532d;
        font-size: 30px;
        font-weight: 850;
        margin-top: 6px;
    }

    .mw-kpi-note {
        color: #94a3b8;
        font-size: 11px;
        margin-top: 4px;
    }

    /* ---------------------------------------------------------------
       ALERT BANNER
    --------------------------------------------------------------- */

    .mw-alert {
        border-radius: 15px;
        padding: 15px 18px;
        margin: 16px 0 20px 0;
        border: 1px solid #fecaca;
        background: linear-gradient(
            90deg,
            #fff1f2,
            #fff7f7
        );
        color: #991b1b;
    }

    .mw-alert-title {
        font-weight: 800;
        font-size: 15px;
    }

    .mw-alert-text {
        font-size: 13px;
        margin-top: 4px;
    }

    /* ---------------------------------------------------------------
       PRIORITY CARD
    --------------------------------------------------------------- */

    .mw-priority {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 5px 16px rgba(15,23,42,0.05);
        margin-bottom: 8px;
    }

    .mw-priority-id {
        font-size: 18px;
        font-weight: 800;
        color: #14532d;
    }

    .mw-priority-risk {
        font-size: 24px;
        font-weight: 850;
        margin-top: 6px;
    }

    .mw-priority-label {
        color: #64748b;
        font-size: 12px;
    }

    /* ---------------------------------------------------------------
       RISK BADGES
    --------------------------------------------------------------- */

    .risk-high {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 800;
    }

    .risk-medium {
        display: inline-block;
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid #fed7aa;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 800;
    }

    .risk-low {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 800;
    }

    /* ---------------------------------------------------------------
       INFO CARDS
    --------------------------------------------------------------- */

    .mw-info {
        background: white;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #16a34a;
        border-radius: 12px;
        padding: 17px 19px;
        box-shadow: 0 4px 15px rgba(15,23,42,0.04);
        margin: 8px 0;
    }

    .mw-info-title {
        color: #14532d;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .mw-info-text {
        color: #475569;
        font-size: 13px;
        line-height: 1.55;
    }

    /* ---------------------------------------------------------------
       MODEL STATUS
    --------------------------------------------------------------- */

    .mw-status {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 9px 13px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: 650;
    }

    /* ---------------------------------------------------------------
       FOOTER
    --------------------------------------------------------------- */

    .mw-footer {
        margin-top: 35px;
        padding: 25px;
        text-align: center;
        color: #64748b;
        border-top: 1px solid #e2e8f0;
        font-size: 12px;
    }

    .mw-footer-title {
        color: #14532d;
        font-size: 16px;
        font-weight: 800;
    }

    /* ---------------------------------------------------------------
       TABS
    --------------------------------------------------------------- */

    button[data-baseweb="tab"] {
        font-weight: 650 !important;
    }

    /* ---------------------------------------------------------------
       DATAFRAME
    --------------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# TEXT - ENGLISH / TAMIL
# =====================================================================

TEXT = {

    "English": {

        "title":
            "🐄 MastiWatch: Mastitis Early-Warning System",

        "sub":
            "AI forecasting of bovine mastitis 3-5 days before symptoms appear",

        "lang":
            "Language / மொழி",

        "settings":
            "Settings",

        "day":
            "Select day",

        "breed":
            "Breed filter",

        "all":
            "All breeds",

        "thr":
            "Alert thresholds",

        "thr_high":
            "HIGH risk from",

        "thr_med":
            "MEDIUM risk from",

        "tabs": [
            "🏠 Herd overview",
            "🐄 Cow detail",
            "🧪 What-if simulator",
            "📊 Model performance",
            "ℹ️ About"
        ],

        "herd":
            "Herd risk overview",

        "cows":
            "Cows",

        "high":
            "HIGH risk",

        "med":
            "MEDIUM risk",

        "low":
            "LOW risk",

        "dist":
            "Risk level split",

        "bybreed":
            "Average risk by breed",

        "table":
            "Cows ranked by risk",

        "download":
            "⬇️ Download alert list (CSV)",

        "top_n":
            "Rows to show",

        "pick":
            "Select cow",

        "risk":
            "Mastitis risk (next 5 days)",

        "why":
            "Why is this cow flagged?",

        "why_help":
            "Red bars push risk UP, green bars push risk DOWN "
            "(XGBoost feature contributions).",

        "trend":
            "Trends for this cow",

        "prob":
            "Risk over time",

        "profile":
            "Cow profile",

        "actions": {

            "HIGH":
                "Do a CMT test today, check milking hygiene, "
                "isolate milk, and call the vet.",

            "MEDIUM":
                "Monitor closely, re-check milk tomorrow, "
                "and keep the udder clean.",

            "LOW":
                "Routine care. Continue regular milking hygiene.",
        },

        "no_data":
            "No data for this cow on or before the selected day.",

        "whatif":
            "What-if simulator",

        "whatif_help":
            "Change the readings of the selected cow and "
            "see how the risk responds.",

        "whatif_cow":
            "Starting from cow",

        "reset":
            "Original risk",

        "new":
            "Simulated risk",

        "perf":
            "Model performance on the demo dataset",

        "perf_note":
            (
                "The model file was trained on this same simulated dataset, "
                "so the numbers below are optimistic. The held-out results "
                "reported in the project deck are ROC-AUC 0.96, recall 75%, "
                "precision 0.33 and about 4 days of early warning."
            ),

        "reported":
            "Reported in project deck (held-out test)",

        "this_data":
            "Computed on the demo dataset (in-sample)",

        "lead":
            "Average early-warning lead time (days)",

        "importance":
            "Most important features",

        "note":
            "Demo uses simulated data based on published ranges. "
            "Retrain on real farm data for deployment.",

        "about_h":
            "About MastiWatch",
    },


    "தமிழ்": {

        "title":
            "🐄 MastiWatch: மடிவீக்க நோய் முன்னெச்சரிக்கை அமைப்பு",

        "sub":
            "அறிகுறிகள் தெரிவதற்கு 3-5 நாட்களுக்கு முன்பே AI எச்சரிக்கை",

        "lang":
            "Language / மொழி",

        "settings":
            "அமைப்புகள்",

        "day":
            "நாளைத் தேர்ந்தெடுக்கவும்",

        "breed":
            "இனம் வடிகட்டி",

        "all":
            "அனைத்து இனங்கள்",

        "thr":
            "எச்சரிக்கை எல்லைகள்",

        "thr_high":
            "அதிக ஆபத்து தொடக்கம்",

        "thr_med":
            "நடுத்தர ஆபத்து தொடக்கம்",

        "tabs": [
            "🏠 மந்தை மேலோட்டம்",
            "🐄 மாட்டின் விவரம்",
            "🧪 என்ன நடந்தால்?",
            "📊 மாதிரி செயல்திறன்",
            "ℹ️ பற்றி"
        ],

        "herd":
            "மந்தை ஆபத்து நிலை",

        "cows":
            "மாடுகள்",

        "high":
            "அதிக ஆபத்து",

        "med":
            "நடுத்தர ஆபத்து",

        "low":
            "குறைந்த ஆபத்து",

        "dist":
            "ஆபத்து நிலை பகிர்வு",

        "bybreed":
            "இனம் வாரியாக சராசரி ஆபத்து",

        "table":
            "ஆபத்து அடிப்படையில் மாடுகள்",

        "download":
            "⬇️ எச்சரிக்கைப் பட்டியல் (CSV)",

        "top_n":
            "காட்ட வேண்டிய வரிசைகள்",

        "pick":
            "மாட்டைத் தேர்ந்தெடுக்கவும்",

        "risk":
            "மடிவீக்க ஆபத்து (அடுத்த 5 நாட்கள்)",

        "why":
            "இந்த மாடு ஏன் எச்சரிக்கப்பட்டது?",

        "why_help":
            "சிவப்பு பட்டை ஆபத்தை உயர்த்தும், "
            "பச்சை பட்டை குறைக்கும்.",

        "trend":
            "இந்த மாட்டின் போக்கு",

        "prob":
            "காலப்போக்கில் ஆபத்து",

        "profile":
            "மாட்டின் சுயவிவரம்",

        "actions": {

            "HIGH":
                "இன்றே CMT சோதனை செய்யுங்கள், "
                "சுத்தத்தைச் சரிபார்த்து, கால்நடை மருத்துவரை அழைக்கவும்.",

            "MEDIUM":
                "கவனமாகக் கண்காணித்து, நாளை மீண்டும் "
                "பால் சோதிக்கவும்.",

            "LOW":
                "வழக்கமான பராமரிப்பைத் தொடரவும்.",
        },

        "no_data":
            "தேர்ந்தெடுத்த நாளுக்கு முன் இந்த மாட்டிற்குத் தரவு இல்லை.",

        "whatif":
            "என்ன நடந்தால்? சிமுலேட்டர்",

        "whatif_help":
            "மாட்டின் அளவீடுகளை மாற்றி ஆபத்து எப்படி "
            "மாறுகிறது என்று பாருங்கள்.",

        "whatif_cow":
            "தொடங்கும் மாடு",

        "reset":
            "அசல் ஆபத்து",

        "new":
            "சிமுலேட் ஆபத்து",

        "perf":
            "டெமோ தரவில் மாதிரி செயல்திறன்",

        "perf_note":
            (
                "மாதிரி இதே உருவகப்படுத்தப்பட்ட தரவில் பயிற்சி பெற்றது; "
                "எனவே இங்குள்ள எண்கள் அதிகமாகத் தெரியும். "
                "திட்ட விளக்கக்காட்சியில் ROC-AUC 0.96, recall 75%, "
                "precision 0.33, சுமார் 4 நாள் முன்னெச்சரிக்கை."
            ),

        "reported":
            "திட்ட அறிக்கையில் (சோதனைத் தரவு)",

        "this_data":
            "டெமோ தரவில் கணக்கிடப்பட்டது",

        "lead":
            "சராசரி முன்னெச்சரிக்கை நாட்கள்",

        "importance":
            "முக்கிய காரணிகள்",

        "note":
            "இது உருவகப்படுத்தப்பட்ட தரவு. "
            "உண்மையான பண்ணை தரவில் மீண்டும் பயிற்சி அளிக்க வேண்டும்.",

        "about_h":
            "MastiWatch பற்றி",
    },
}


# =====================================================================
# FEATURE NAMES
# =====================================================================

NICE = {

    "scc_ma3":
        "Somatic cell count (3-day avg)",

    "scc_ma7":
        "Somatic cell count (7-day avg)",

    "scc_log":
        "Somatic cell count (log)",

    "scc_dev":
        "SCC vs cow's own normal",

    "scc_chg3":
        "SCC change (3 days)",

    "conductivity":
        "Milk conductivity",

    "conductivity_ma3":
        "Conductivity (3-day avg)",

    "conductivity_ma7":
        "Conductivity (7-day avg)",

    "conductivity_dev":
        "Conductivity vs cow's normal",

    "conductivity_chg3":
        "Conductivity change (3 days)",

    "yield_l":
        "Milk yield",

    "yield_l_ma3":
        "Yield (3-day avg)",

    "yield_l_ma7":
        "Yield (7-day avg)",

    "yield_l_dev":
        "Yield vs cow's normal",

    "yield_l_chg3":
        "Yield change (3 days)",

    "body_temp":
        "Body temperature",

    "body_temp_ma3":
        "Body temp (3-day avg)",

    "body_temp_ma7":
        "Body temp (7-day avg)",

    "body_temp_dev":
        "Body temp vs normal",

    "body_temp_chg3":
        "Body temp change",

    "hand_milking":
        "Hand milking",

    "hygiene":
        "Hygiene score",

    "parity":
        "Number of calvings",

    "humidity":
        "Humidity",

    "thi":
        "Heat stress (THI)",

    "thi_ma3":
        "Heat stress (3-day avg)",

    "temp":
        "Air temperature",

    "dim":
        "Days in milk",

    "breed_code":
        "Breed",
}


# =====================================================================
# MODEL LOADING
# =====================================================================

@st.cache_resource
def load_model():

    model_file = BASE / "mastitis_model.pkl"

    if not model_file.exists():

        st.error(
            "❌ mastitis_model.pkl was not found.\n\n"
            "Please keep mastitis_model.pkl in the same folder as app.py."
        )

        st.stop()

    bundle = joblib.load(model_file)

    return bundle["model"], bundle["features"]


# =====================================================================
# DATA LOADING
# =====================================================================

@st.cache_data
def load_scored_data():

    model, feats = load_model()

    data_file = BASE / "mastitis_features.csv"

    if not data_file.exists():

        st.error(
            "❌ mastitis_features.csv was not found.\n\n"
            "Please keep mastitis_features.csv in the same folder as app.py."
        )

        st.stop()

    df = pd.read_csv(data_file)

    missing = [
        f for f in feats
        if f not in df.columns
    ]

    if missing:

        st.error(
            "❌ These model features are missing from the CSV:"
        )

        for feature in missing:
            st.write(f"- `{feature}`")

        st.stop()

    df["risk"] = model.predict_proba(
        df[feats]
    )[:, 1]

    return df


# =====================================================================
# RISK BAND
# =====================================================================

def band(p, high, med):

    if p >= high:
        return "HIGH"

    elif p >= med:
        return "MEDIUM"

    return "LOW"


# =====================================================================
# XGBOOST CONTRIBUTIONS
# =====================================================================

def contributions(model, feats, row_df):

    try:

        import xgboost as xgb

        d = xgb.DMatrix(
            row_df[feats]
        )

        c = model.get_booster().predict(
            d,
            pred_contribs=True
        )[0][:-1]

        return pd.Series(
            c,
            index=feats
        )

    except Exception:

        return None


# =====================================================================
# LEAD TIME
# =====================================================================

def lead_time(df, thr):

    out = []

    for _, g in df[
        df["onset_day"] >= 0
    ].groupby("cow_id"):

        onset = g["onset_day"].iloc[0]

        pre = g[
            (g["day"] < onset)
            &
            (g["day"] >= onset - 5)
            &
            (g["risk"] >= thr)
        ]

        if len(pre):

            out.append(
                onset - pre["day"].min()
            )

    return (
        float(np.mean(out))
        if out
        else float("nan"),
        len(out)
    )


# =====================================================================
# GAUGE
# =====================================================================

def gauge(p, high, med):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={
                "suffix": " %",
                "font": {
                    "size": 34,
                    "color": DARK_GREEN
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },

                "bar": {
                    "color": DARK_GREEN
                },

                "steps": [

                    {
                        "range": [
                            0,
                            med * 100
                        ],
                        "color": "#dcfce7"
                    },

                    {
                        "range": [
                            med * 100,
                            high * 100
                        ],
                        "color": "#fef3c7"
                    },

                    {
                        "range": [
                            high * 100,
                            100
                        ],
                        "color": "#fee2e2"
                    },
                ]
            },
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(
            l=10,
            r=10,
            t=15,
            b=0
        )
    )

    return fig


# =====================================================================
# LINE CHART
# =====================================================================

def line(
    df,
    y,
    title,
    colour=BLUE,
    hlines=None,
    day=None,
    yfmt=None
):

    fig = go.Figure(
        go.Scatter(
            x=df["day"],
            y=df[y],
            mode="lines+markers",
            line=dict(
                color=colour,
                width=3
            ),
            marker=dict(
                size=6
            )
        )
    )

    for val, col, txt in (
        hlines or []
    ):

        fig.add_hline(
            y=val,
            line_dash="dash",
            line_color=col,
            annotation_text=txt
        )

    if day is not None:

        fig.add_vline(
            x=day,
            line_dash="dot",
            line_color="grey"
        )

    fig.update_layout(
        title=title,
        height=280,
        margin=dict(
            l=10,
            r=10,
            t=45,
            b=10
        ),
        xaxis_title="Day",
        yaxis_tickformat=yfmt,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# =====================================================================
# LOAD APPLICATION
# =====================================================================

lang = st.sidebar.radio(
    "Language / மொழி",
    list(TEXT)
)

T = TEXT[lang]

model, FEATURES = load_model()

DF = load_scored_data()


# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        padding:12px 0 5px 0;
    ">
        <div style="
            font-size:42px;
        ">🐄</div>

        <div style="
            font-size:22px;
            font-weight:800;
            color:#14532d;
        ">
            MastiWatch
        </div>

        <div style="
            font-size:11px;
            color:#64748b;
            margin-top:3px;
        ">
            AI Mastitis Early Warning
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

st.sidebar.header(
    T["settings"]
)

days = sorted(
    DF["day"].unique()
)

day = st.sidebar.select_slider(
    T["day"],
    options=days,
    value=days[len(days) // 2]
)

breeds = [
    T["all"]
] + sorted(
    DF["breed"].unique()
)

breed_sel = st.sidebar.selectbox(
    T["breed"],
    breeds
)

st.sidebar.markdown(
    "**" + T["thr"] + "**"
)

HIGH = st.sidebar.slider(
    T["thr_high"],
    0.30,
    0.95,
    0.60,
    0.05
)

MED = st.sidebar.slider(
    T["thr_med"],
    0.05,
    float(HIGH) - 0.05,
    min(
        0.30,
        HIGH - 0.05
    ),
    0.05
)

st.sidebar.divider()

st.sidebar.markdown(
    f"""
    <div class="mw-status">
        🟢 AI Model Loaded<br>
        📊 {len(FEATURES)} predictive features<br>
        🐄 {DF['cow_id'].nunique()} cows<br>
        📅 {len(days)} observation days
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.info(
    T["note"]
)


# =====================================================================
# RISK BANDS
# =====================================================================

DF["band"] = DF["risk"].apply(
    lambda p: band(
        p,
        HIGH,
        MED
    )
)


# =====================================================================
# FILTER DATA
# =====================================================================

VIEW = (
    DF
    if breed_sel == T["all"]
    else DF[
        DF["breed"] == breed_sel
    ]
)


# =====================================================================
# HERO HEADER
# =====================================================================

st.markdown(
    f"""
    <div class="mw-hero">
        <div class="mw-hero-title">
            🐄 MastiWatch
        </div>

        <div class="mw-hero-subtitle">
            AI-Based Early Warning System for Bovine Mastitis
        </div>

        <div class="mw-hero-description">
            {T["sub"]}
        </div>

        <div class="mw-hero-pill">
            🤖 XGBoost &nbsp; • &nbsp;
            🧠 Explainable AI &nbsp; • &nbsp;
            📊 5-Day Forecast &nbsp; • &nbsp;
            🇮🇳 Built for Indian Dairy Farms
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================================
# LATEST READING PER COW
# =====================================================================

latest = (
    VIEW[
        VIEW["day"] <= day
    ]
    .sort_values("day")
    .groupby("cow_id")
    .tail(1)
)

latest = latest[
    latest["day"] == day
]

today = latest.sort_values(
    "risk",
    ascending=False
)


# =====================================================================
# TABS
# =====================================================================

tab_herd, tab_cow, tab_what, tab_perf, tab_about = st.tabs(
    T["tabs"]
)


# =====================================================================
# TAB 1 — HERD OVERVIEW
# =====================================================================

with tab_herd:

    st.markdown(
        f"""
        <div class="mw-section-title">
            🏠 {T["herd"]}
        </div>

        <div class="mw-section-subtitle">
            Real-time AI risk prioritization for the selected day
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------
    # KPI VALUES
    # ---------------------------------------------------------------

    total_cows = len(today)

    high_count = int(
        (today.band == "HIGH").sum()
    )

    medium_count = int(
        (today.band == "MEDIUM").sum()
    )

    low_count = int(
        (today.band == "LOW").sum()
    )

    average_risk = (
        today["risk"].mean()
        if not today.empty
        else 0
    )

    # ---------------------------------------------------------------
    # KPI CARDS
    # ---------------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="mw-kpi">
                <div class="mw-kpi-label">
                    🐄 {T["cows"]}
                </div>

                <div class="mw-kpi-value">
                    {total_cows}
                </div>

                <div class="mw-kpi-note">
                    Monitored on Day {day}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="mw-kpi">
                <div class="mw-kpi-label">
                    🔴 {T["high"]}
                </div>

                <div class="mw-kpi-value"
                     style="color:#b91c1c;">
                    {high_count}
                </div>

                <div class="mw-kpi-note">
                    Immediate attention
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="mw-kpi">
                <div class="mw-kpi-label">
                    🟠 {T["med"]}
                </div>

                <div class="mw-kpi-value"
                     style="color:#c2410c;">
                    {medium_count}
                </div>

                <div class="mw-kpi-note">
                    Increased monitoring
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="mw-kpi">
                <div class="mw-kpi-label">
                    🟢 {T["low"]}
                </div>

                <div class="mw-kpi-value"
                     style="color:#15803d;">
                    {low_count}
                </div>

                <div class="mw-kpi-note">
                    Routine monitoring
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------------------------------------
    # ALERT BANNER
    # ---------------------------------------------------------------

    if high_count > 0:

        st.markdown(
            f"""
            <div class="mw-alert">
                <div class="mw-alert-title">
                    🚨 {high_count} high-risk cow(s) detected
                </div>

                <div class="mw-alert-text">
                    Prioritize these cows for closer monitoring,
                    CMT testing and veterinary assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.success(
            "🟢 No high-risk cows detected at the selected threshold."
        )

    # ---------------------------------------------------------------
    # TOP PRIORITY COWS
    # ---------------------------------------------------------------

    if not today.empty:

        st.markdown(
            "### 🚨 Top Priority Cows"
        )

        priority = today.head(3)

        cols = st.columns(
            len(priority)
        )

        for col, (_, cow_row) in zip(
            cols,
            priority.iterrows()
        ):

            with col:

                risk_percent = (
                    cow_row["risk"] * 100
                )

                cow_band = cow_row["band"]

                border_color = BAND_COLOUR[
                    cow_band
                ]

                st.markdown(
                    f"""
                    <div class="mw-priority"
                         style="border-top:4px solid {border_color};">

                        <div class="mw-priority-id">
                            🐄 Cow {int(cow_row["cow_id"])}
                        </div>

                        <div class="mw-priority-risk"
                             style="color:{border_color};">
                            {risk_percent:.1f}%
                        </div>

                        <div class="mw-priority-label">
                            Predicted mastitis risk
                        </div>

                        <br>

                        <span class="risk-{cow_band.lower()}">
                            {BAND_ICON[cow_band]} {cow_band}
                        </span>

                        <br><br>

                        <div class="mw-priority-label">
                            Breed: <b>{cow_row["breed"]}</b>
                        </div>

                        <div class="mw-priority-label">
                            Milk yield: <b>{cow_row["yield_l"]:.2f} L</b>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ---------------------------------------------------------------
    # CHARTS
    # ---------------------------------------------------------------

    if today.empty:

        st.warning(
            T["no_data"]
        )

    else:

        a, b = st.columns(2)

        # Risk distribution

        counts = (
            today["band"]
            .value_counts()
            .reindex(
                [
                    "HIGH",
                    "MEDIUM",
                    "LOW"
                ]
            )
            .fillna(0)
        )

        pie = go.Figure(
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.58,
                marker_colors=[
                    BAND_COLOUR[k]
                    for k in counts.index
                ],
                textinfo="label+percent"
            )
        )

        pie.update_layout(
            title=T["dist"],
            height=330,
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            ),
            paper_bgcolor="white"
        )

        a.plotly_chart(
            pie,
            **STRETCH
        )

        # Breed risk

        br = (
            today
            .groupby("breed")["risk"]
            .mean()
            .mul(100)
            .sort_values()
        )

        bar = go.Figure(
            go.Bar(
                x=br.values,
                y=br.index,
                orientation="h",
                marker_color=BLUE,
                text=[
                    f"{v:.1f}%"
                    for v in br.values
                ],
                textposition="outside"
            )
        )

        bar.update_layout(
            title=T["bybreed"],
            height=330,
            xaxis_title="Average risk (%)",
            margin=dict(
                l=10,
                r=35,
                t=50,
                b=10
            ),
            paper_bgcolor="white"
        )

        b.plotly_chart(
            bar,
            **STRETCH
        )

        # -----------------------------------------------------------
        # TABLE
        # -----------------------------------------------------------

        st.markdown(
            "### 📋 " + T["table"]
        )

        n = st.slider(
            T["top_n"],
            5,
            min(
                50,
                len(today)
            ),
            min(
                15,
                len(today)
            )
        )

        show = today[
            [
                "cow_id",
                "breed",
                "risk",
                "band",
                "yield_l",
                "conductivity",
                "scc",
                "body_temp"
            ]
        ].head(n).copy()

        show["band"] = show[
            "band"
        ].map(
            lambda k:
            f"{BAND_ICON[k]} {k}"
        )

        show.columns = [
            "Cow",
            "Breed",
            "Risk",
            "Level",
            "Yield (L)",
            "Conductivity",
            "SCC (cells/mL)",
            "Body temp (°C)"
        ]

        st.dataframe(
            show.round(2),
            **STRETCH,
            hide_index=True,
            column_config={
                "Risk":
                    st.column_config.ProgressColumn(
                        "Risk",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f"
                    )
            }
        )

        # -----------------------------------------------------------
        # DOWNLOAD
        # -----------------------------------------------------------

        st.download_button(
            T["download"],
            today
            .drop(
                columns=["band"]
            )
            .to_csv(
                index=False
            )
            .encode(),

            f"mastiwatch_alerts_day{day}.csv",

            "text/csv"
        )


# =====================================================================
# TAB 2 — COW DETAIL
# =====================================================================

with tab_cow:

    st.markdown(
        f"""
        <div class="mw-section-title">
            🐄 {T["tabs"][1]}
        </div>

        <div class="mw-section-subtitle">
            Individual cow risk, trends and explainable AI
        </div>
        """,
        unsafe_allow_html=True
    )

    cow_ids = sorted(
        VIEW["cow_id"].unique()
    )

    default = (
        int(today.iloc[0]["cow_id"])
        if len(today)
        else cow_ids[0]
    )

    cow = st.selectbox(
        T["pick"],
        cow_ids,
        index=cow_ids.index(
            default
        )
    )

    cow_df = (
        DF[
            DF["cow_id"] == cow
        ]
        .sort_values("day")
    )

    now = cow_df[
        cow_df["day"] <= day
    ].tail(1)

    if now.empty:

        st.warning(
            T["no_data"]
        )

    else:

        r = now.iloc[0]

        p = float(
            r["risk"]
        )

        b = band(
            p,
            HIGH,
            MED
        )

        left, right = st.columns(
            [1, 2]
        )

        # -----------------------------------------------------------
        # LEFT - RISK
        # -----------------------------------------------------------

        with left:

            st.caption(
                T["risk"]
            )

            st.plotly_chart(
                gauge(
                    p,
                    HIGH,
                    MED
                ),
                **STRETCH
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    margin-bottom:12px;
                ">
                    <span class="risk-{b.lower()}">
                        {BAND_ICON[b]} {b} RISK
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            if b == "HIGH":

                st.error(
                    T["actions"][b]
                )

            elif b == "MEDIUM":

                st.warning(
                    T["actions"][b]
                )

            else:

                st.success(
                    T["actions"][b]
                )

            st.markdown(
                "### 👤 " + T["profile"]
            )

            st.markdown(
                f"""
                <div class="mw-info">

                    <div class="mw-info-text">

                        <b>🐄 Cow ID:</b>
                        {int(r["cow_id"])}
                        <br><br>

                        <b>Breed:</b>
                        {r["breed"]}
                        <br>

                        <b>Calvings:</b>
                        {int(r["parity"])}
                        <br>

                        <b>Days in milk:</b>
                        {int(r["dim"])}
                        <br>

                        <b>Hand milking:</b>
                        {"Yes" if r["hand_milking"] else "No"}
                        <br>

                        <b>Hygiene score:</b>
                        {int(r["hygiene"])}/5

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # -----------------------------------------------------------
        # RIGHT - AI EXPLANATION
        # -----------------------------------------------------------

        with right:

            st.markdown(
                "### 🧠 " + T["why"]
            )

            st.caption(
                T["why_help"]
            )

            c = contributions(
                model,
                FEATURES,
                now
            )

            if c is not None:

                top = (
                    c
                    .reindex(
                        c.abs()
                        .sort_values(
                            ascending=False
                        )
                        .index
                    )
                    .head(8)[::-1]
                )

                fig = go.Figure(
                    go.Bar(
                        x=top.values,
                        y=[
                            NICE.get(
                                f,
                                f
                            )
                            for f in top.index
                        ],
                        orientation="h",
                        marker_color=[
                            RED if v > 0
                            else GREEN
                            for v in top.values
                        ]
                    )
                )

                fig.update_layout(
                    height=370,
                    margin=dict(
                        l=10,
                        r=10,
                        t=10,
                        b=10
                    ),
                    xaxis_title=
                        "Effect on risk (log-odds)",
                    plot_bgcolor="white"
                )

                st.plotly_chart(
                    fig,
                    **STRETCH
                )

            else:

                st.info(
                    "Explanation not available for this model type."
                )

        # -----------------------------------------------------------
        # RISK TREND
        # -----------------------------------------------------------

        st.markdown(
            "### 📈 " + T["prob"]
        )

        st.plotly_chart(
            line(
                cow_df,
                "risk",
                T["prob"],
                RED,
                day=day,
                hlines=[
                    (
                        HIGH,
                        RED,
                        "HIGH"
                    ),
                    (
                        MED,
                        ORANGE,
                        "MEDIUM"
                    )
                ],
                yfmt=".0%"
            ),
            **STRETCH
        )

        # -----------------------------------------------------------
        # HEALTH TRENDS
        # -----------------------------------------------------------

        st.markdown(
            "### 📊 " + T["trend"]
        )

        t1, t2 = st.columns(2)

        t1.plotly_chart(
            line(
                cow_df,
                "yield_l",
                "Milk yield (L)",
                BLUE,
                day=day
            ),
            **STRETCH
        )

        t2.plotly_chart(
            line(
                cow_df,
                "conductivity",
                "Milk conductivity (mS/cm)",
                ORANGE,
                day=day
            ),
            **STRETCH
        )

        t3, t4 = st.columns(2)

        t3.plotly_chart(
            line(
                cow_df,
                "scc",
                "Somatic cell count (cells/mL)",
                RED,
                day=day
            ),
            **STRETCH
        )

        t4.plotly_chart(
            line(
                cow_df,
                "body_temp",
                "Body temperature (°C)",
                GREEN,
                day=day
            ),
            **STRETCH
        )


# =====================================================================
# TAB 3 — WHAT IF SIMULATOR
# =====================================================================

with tab_what:

    st.markdown(
        f"""
        <div class="mw-section-title">
            🧪 {T["whatif"]}
        </div>

        <div class="mw-section-subtitle">
            Explore how changes in cow-level parameters
            may affect predicted mastitis risk.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "💡 " + T["whatif_help"]
    )

    ids = sorted(
        VIEW["cow_id"].unique()
    )

    wcow = st.selectbox(
        T["whatif_cow"],
        ids,
        index=(
            ids.index(
                int(today.iloc[0]["cow_id"])
            )
            if len(today)
            else 0
        ),
        key="wcow"
    )

    base_df = (
        DF[
            (DF["cow_id"] == wcow)
            &
            (DF["day"] <= day)
        ]
        .sort_values("day")
        .tail(1)
    )

    if base_df.empty:

        st.warning(
            T["no_data"]
        )

    else:

        base = base_df.iloc[0]

        st.markdown(
            "### 🎛️ Adjust Parameters"
        )

        s1, s2, s3 = st.columns(3)

        scc_log = s1.slider(
            "SCC (log scale)",
            8.0,
            15.0,
            float(
                base["scc_log"]
            ),
            0.1
        )

        scc_dev = s1.slider(
            "SCC vs cow's normal",
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

        cond = s2.slider(
            "Conductivity (mS/cm)",
            3.5,
            7.5,
            float(
                base["conductivity"]
            ),
            0.05
        )

        cond_dev = s2.slider(
            "Conductivity vs normal",
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

        y_dev = s3.slider(
            "Yield vs normal (L)",
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

        btemp = s3.slider(
            "Body temp (°C)",
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

        h1, h2 = st.columns(2)

        hyg = h1.slider(
            "Hygiene score",
            1,
            5,
            int(
                base["hygiene"]
            )
        )

        hand = h2.radio(
            "Hand milking",
            [0, 1],
            index=int(
                base["hand_milking"]
            ),
            format_func=lambda v:
                "Yes" if v else "No",
            horizontal=True
        )

        new = base_df.copy()

        new["scc_log"] = scc_log
        new["scc_dev"] = scc_dev

        new["conductivity"] = cond
        new["conductivity_dev"] = cond_dev

        new["yield_l_dev"] = y_dev

        new["body_temp"] = btemp

        new["hygiene"] = hyg
        new["hand_milking"] = hand

        new_p = float(
            model.predict_proba(
                new[FEATURES]
            )[:, 1][0]
        )

        old_p = float(
            base["risk"]
        )

        difference = (
            new_p - old_p
        )

        # -----------------------------------------------------------
        # SIMULATION RESULT
        # -----------------------------------------------------------

        st.markdown(
            "### 📊 Simulation Result"
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            T["reset"],
            f"{old_p * 100:.0f}%"
        )

        m2.metric(
            T["new"],
            f"{new_p * 100:.0f}%"
        )

        m3.metric(
            "Risk change",
            f"{difference * 100:+.0f} pts",
            delta_color="inverse"
        )

        nb = band(
            new_p,
            HIGH,
            MED
        )

        if nb == "HIGH":

            st.error(
                f"{BAND_ICON[nb]} {nb}: "
                f"{T['actions'][nb]}"
            )

        elif nb == "MEDIUM":

            st.warning(
                f"{BAND_ICON[nb]} {nb}: "
                f"{T['actions'][nb]}"
            )

        else:

            st.success(
                f"{BAND_ICON[nb]} {nb}: "
                f"{T['actions'][nb]}"
            )


# =====================================================================
# TAB 4 — MODEL PERFORMANCE
# =====================================================================

with tab_perf:

    st.markdown(
        f"""
        <div class="mw-section-title">
            📊 {T["perf"]}
        </div>

        <div class="mw-section-subtitle">
            Evaluation and explainability of the MastiWatch AI model
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "⚠️ " + T["perf_note"]
    )

    y = DF["label"]

    s = DF["risk"]

    pred = (
        s >= MED
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred
        )
        .ravel()
    )

    rec = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    prec = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    lt, n_cows = lead_time(
        DF,
        MED
    )

    # ---------------------------------------------------------------
    # REPORTED RESULTS
    # ---------------------------------------------------------------

    st.markdown(
        "### 🏆 Project Report Results"
    )

    k = st.columns(4)

    k[0].metric(
        "ROC-AUC",
        "0.96"
    )

    k[1].metric(
        "Recall",
        "75%"
    )

    k[2].metric(
        "Precision",
        "0.33"
    )

    k[3].metric(
        "Early warning",
        "~4 days"
    )

    # ---------------------------------------------------------------
    # CURRENT DATA
    # ---------------------------------------------------------------

    st.markdown(
        "### 📌 Current Demo Dataset"
    )

    k = st.columns(4)

    k[0].metric(
        "ROC-AUC",
        f"{roc_auc_score(y, s):.2f}"
    )

    k[1].metric(
        "Recall",
        f"{rec:.0%}"
    )

    k[2].metric(
        "Precision",
        f"{prec:.2f}"
    )

    k[3].metric(
        T["lead"],
        f"{lt:.1f}"
    )

    # ---------------------------------------------------------------
    # ROC / PR
    # ---------------------------------------------------------------

    a, b = st.columns(2)

    fpr, tpr, _ = roc_curve(
        y,
        s
    )

    roc = go.Figure()

    roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            name="MastiWatch",
            line=dict(
                color=BLUE,
                width=3
            )
        )
    )

    roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Random",
            line=dict(
                dash="dash",
                color="grey"
            )
        )
    )

    roc.update_layout(
        title="ROC Curve",
        height=350,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        ),
        plot_bgcolor="white"
    )

    a.plotly_chart(
        roc,
        **STRETCH
    )

    pr, rc, _ = (
        precision_recall_curve(
            y,
            s
        )
    )

    prf = go.Figure(
        go.Scatter(
            x=rc,
            y=pr,
            line=dict(
                color=ORANGE,
                width=3
            )
        )
    )

    prf.update_layout(
        title="Precision-Recall Curve",
        height=350,
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        ),
        plot_bgcolor="white"
    )

    b.plotly_chart(
        prf,
        **STRETCH
    )

    # ---------------------------------------------------------------
    # CONFUSION MATRIX / FEATURE IMPORTANCE
    # ---------------------------------------------------------------

    c, d = st.columns(2)

    cm = go.Figure(
        go.Heatmap(
            z=[
                [tn, fp],
                [fn, tp]
            ],
            x=[
                "Pred: healthy",
                "Pred: at risk"
            ],
            y=[
                "Actual: healthy",
                "Actual: at risk"
            ],
            colorscale="Greens",
            text=[
                [tn, fp],
                [fn, tp]
            ],
            texttemplate="%{text}",
            showscale=False
        )
    )

    cm.update_layout(
        title="Confusion Matrix",
        height=350,
        yaxis_autorange="reversed",
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        )
    )

    c.plotly_chart(
        cm,
        **STRETCH
    )

    imp = (
        pd.Series(
            model.feature_importances_,
            index=FEATURES
        )
        .sort_values()
        .tail(10)
    )

    fi = go.Figure(
        go.Bar(
            x=imp.values,
            y=[
                NICE.get(
                    f,
                    f
                )
                for f in imp.index
            ],
            orientation="h",
            marker_color=BLUE
        )
    )

    fi.update_layout(
        title=T["importance"],
        height=350,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        ),
        plot_bgcolor="white"
    )

    d.plotly_chart(
        fi,
        **STRETCH
    )


# =====================================================================
# TAB 5 — ABOUT
# =====================================================================

with tab_about:

    st.markdown(
        f"""
        <div class="mw-section-title">
            ℹ️ {T["about_h"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------
    # PROJECT INTRO
    # ---------------------------------------------------------------

    st.markdown(
        """
        <div class="mw-info">

            <div class="mw-info-title">
                🐄 What is MastiWatch?
            </div>

            <div class="mw-info-text">

                <b>MastiWatch</b> is a software-only AI system
                that turns routinely recorded farm data
                such as milk yield, conductivity, somatic cell
                count, cow details and weather into a
                <b>5-day mastitis risk score</b>.

                <br><br>

                Every cow is classified into
                <b>Low, Medium or High risk</b>,
                allowing farmers and veterinary teams to
                prioritize animals that require closer attention.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------
    # THREE FEATURE CARDS
    # ---------------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="mw-info">

                <div class="mw-info-title">
                    🤖 AI Forecasting
                </div>

                <div class="mw-info-text">
                    XGBoost analyzes multiple cow-level
                    and environmental indicators to
                    estimate future mastitis risk.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="mw-info">

                <div class="mw-info-title">
                    🧠 Explainable AI
                </div>

                <div class="mw-info-text">
                    Feature contributions show why a
                    particular cow has been flagged,
                    supporting transparent decision-making.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="mw-info">

                <div class="mw-info-title">
                    🚨 Early Intervention
                </div>

                <div class="mw-info-text">
                    The dashboard ranks cows by risk and
                    provides suggested monitoring actions
                    for faster intervention.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------------------------------------
    # PIPELINE
    # ---------------------------------------------------------------

    st.markdown(
        "### 🔄 MastiWatch AI Pipeline"
    )

    st.markdown(
        """
        <div class="mw-info">

            <div class="mw-info-text"
                 style="font-size:15px;text-align:center;">

                🐄 <b>Farm Data</b>
                &nbsp; → &nbsp;

                ⚙️ <b>Feature Engineering</b>
                &nbsp; → &nbsp;

                🤖 <b>XGBoost</b>
                &nbsp; → &nbsp;

                🧠 <b>Explainable AI</b>
                &nbsp; → &nbsp;

                🚨 <b>Risk Alert</b>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------
    # PROJECT DETAILS
    # ---------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🇮🇳 Built for Indian Dairy Farms"
        )

        st.markdown(
            """
            - Heat stress / THI
            - Breed information
            - Hand milking
            - Hygiene score
            - Milk yield
            - Conductivity
            - Somatic cell count
            - English / Tamil support
            """
        )

    with col2:

        st.markdown(
            "### 🛠️ Technology Stack"
        )

        st.markdown(
            """
            - Python
            - Pandas
            - NumPy
            - Scikit-learn
            - XGBoost
            - Plotly
            - Streamlit
            - Explainable AI
            """
        )

    # ---------------------------------------------------------------
    # TEAM
    # ---------------------------------------------------------------

    st.markdown(
        "### 👩‍💻 Team Med Sphere"
    )

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
        **STRETCH,
        hide_index=True
    )

    # ---------------------------------------------------------------
    # LIMITATION
    # ---------------------------------------------------------------

    st.markdown(
        """
        <div class="mw-alert"
             style="
                 background:#fff7ed;
                 border-color:#fed7aa;
                 color:#9a3412;
             ">

            <div class="mw-alert-title">
                ⚠️ Research Prototype
            </div>

            <div class="mw-alert-text">
                MastiWatch is a research and hackathon prototype.
                The current demonstration uses simulated data based
                on published ranges. The model should be validated
                using real farm records and veterinary expertise
                before deployment.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================================
# FOOTER
# =====================================================================

st.markdown(
    """
    <div class="mw-footer">

        <div class="mw-footer-title">
            🐄 MastiWatch
        </div>

        <div style="margin-top:5px;">
            AI-Based Predictive Modelling for Early Forecasting
            of Bovine Mastitis
        </div>

        <div style="margin-top:8px;">
            Team Med Sphere • PSNA College of Engineering
            and Technology
        </div>

        <div style="margin-top:12px;">
            Detect early • Act early • Protect herd health
        </div>

        <div style="margin-top:12px;font-size:11px;">
            ⚠️ Not a veterinary diagnostic tool
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

