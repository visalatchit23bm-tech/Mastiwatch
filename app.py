# =====================================================================
# MastiWatch - Mastitis Early-Warning Dashboard (Streamlit)
# Team Med Sphere | PSNA College of Engineering and Technology
#
# Files needed in the SAME folder:
#     app.py, mastitis_model.pkl, mastitis_features.csv
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
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

BASE = Path(__file__).parent

# Streamlit renamed use_container_width -> width="stretch"; support both.
import streamlit as _st
try:
    from packaging.version import Version
    STRETCH = {"width": "stretch"} if Version(_st.__version__) >= Version("1.50") else {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}
st.set_page_config(page_title="MastiWatch | Mastitis Early Warning",
                   page_icon="🐄", layout="wide")

# ------------------------------------------------------------------ colours
RED, ORANGE, GREEN, BLUE = "#d62728", "#ff9f1c", "#2ca02c", "#1f77b4"
BAND_COLOUR = {"HIGH": RED, "MEDIUM": ORANGE, "LOW": GREEN}
BAND_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}

# ------------------------------------------------------------------ text (EN / TA)
TEXT = {
    "English": {
        "title": "🐄 MastiWatch: Mastitis Early-Warning System",
        "sub": "AI forecasting of bovine mastitis 3-5 days before symptoms appear",
        "lang": "Language / மொழி",
        "settings": "Settings",
        "day": "Select day",
        "breed": "Breed filter",
        "all": "All breeds",
        "thr": "Alert thresholds",
        "thr_high": "HIGH risk from",
        "thr_med": "MEDIUM risk from",
        "tabs": ["🏠 Herd overview", "🐄 Cow detail", "🧪 What-if simulator",
                 "📊 Model performance", "ℹ️ About"],
        "herd": "Herd risk overview",
        "cows": "Cows", "high": "HIGH risk", "med": "MEDIUM risk", "low": "LOW risk",
        "dist": "Risk level split", "bybreed": "Average risk by breed",
        "table": "Cows ranked by risk",
        "download": "⬇️ Download alert list (CSV)",
        "top_n": "Rows to show",
        "pick": "Select cow",
        "risk": "Mastitis risk (next 5 days)",
        "why": "Why is this cow flagged?",
        "why_help": "Red bars push risk UP, green bars push risk DOWN (XGBoost feature contributions).",
        "trend": "Trends for this cow",
        "prob": "Risk over time",
        "profile": "Cow profile",
        "actions": {
            "HIGH": "Do a CMT test today, check milking hygiene, isolate milk, and call the vet.",
            "MEDIUM": "Monitor closely, re-check milk tomorrow, and keep the udder clean.",
            "LOW": "Routine care. Continue regular milking hygiene.",
        },
        "no_data": "No data for this cow on or before the selected day.",
        "whatif": "What-if simulator",
        "whatif_help": "Change the readings of the selected cow and see how the risk responds.",
        "whatif_cow": "Starting from cow",
        "reset": "Original risk", "new": "Simulated risk",
        "perf": "Model performance on the demo dataset",
        "perf_note": ("The model file was trained on this same simulated dataset, so the numbers below "
                      "are optimistic. The held-out results reported in the project deck are "
                      "ROC-AUC 0.96, recall 75%, precision 0.33 and about 4 days of early warning."),
        "reported": "Reported in project deck (held-out test)",
        "this_data": "Computed on the demo dataset (in-sample)",
        "lead": "Average early-warning lead time (days)",
        "importance": "Most important features",
        "note": "Demo uses simulated data based on published ranges. Retrain on real farm data for deployment.",
        "about_h": "About MastiWatch",
    },
    "தமிழ்": {
        "title": "🐄 MastiWatch: மடிவீக்க நோய் முன்னெச்சரிக்கை அமைப்பு",
        "sub": "அறிகுறிகள் தெரிவதற்கு 3-5 நாட்களுக்கு முன்பே AI எச்சரிக்கை",
        "lang": "Language / மொழி",
        "settings": "அமைப்புகள்",
        "day": "நாளைத் தேர்ந்தெடுக்கவும்",
        "breed": "இனம் வடிகட்டி",
        "all": "அனைத்து இனங்கள்",
        "thr": "எச்சரிக்கை எல்லைகள்",
        "thr_high": "அதிக ஆபத்து தொடக்கம்",
        "thr_med": "நடுத்தர ஆபத்து தொடக்கம்",
        "tabs": ["🏠 மந்தை மேலோட்டம்", "🐄 மாட்டின் விவரம்", "🧪 என்ன நடந்தால்?",
                 "📊 மாதிரி செயல்திறன்", "ℹ️ பற்றி"],
        "herd": "மந்தை ஆபத்து நிலை",
        "cows": "மாடுகள்", "high": "அதிக ஆபத்து", "med": "நடுத்தர ஆபத்து", "low": "குறைந்த ஆபத்து",
        "dist": "ஆபத்து நிலை பகிர்வு", "bybreed": "இனம் வாரியாக சராசரி ஆபத்து",
        "table": "ஆபத்து அடிப்படையில் மாடுகள்",
        "download": "⬇️ எச்சரிக்கைப் பட்டியல் (CSV)",
        "top_n": "காட்ட வேண்டிய வரிசைகள்",
        "pick": "மாட்டைத் தேர்ந்தெடுக்கவும்",
        "risk": "மடிவீக்க ஆபத்து (அடுத்த 5 நாட்கள்)",
        "why": "இந்த மாடு ஏன் எச்சரிக்கப்பட்டது?",
        "why_help": "சிவப்பு பட்டை ஆபத்தை உயர்த்தும், பச்சை பட்டை குறைக்கும்.",
        "trend": "இந்த மாட்டின் போக்கு",
        "prob": "காலப்போக்கில் ஆபத்து",
        "profile": "மாட்டின் சுயவிவரம்",
        "actions": {
            "HIGH": "இன்றே CMT சோதனை செய்யுங்கள், சுத்தத்தைச் சரிபார்த்து, கால்நடை மருத்துவரை அழைக்கவும்.",
            "MEDIUM": "கவனமாகக் கண்காணித்து, நாளை மீண்டும் பால் சோதிக்கவும்.",
            "LOW": "வழக்கமான பராமரிப்பைத் தொடரவும்.",
        },
        "no_data": "தேர்ந்தெடுத்த நாளுக்கு முன் இந்த மாட்டிற்குத் தரவு இல்லை.",
        "whatif": "என்ன நடந்தால்? சிமுலேட்டர்",
        "whatif_help": "மாட்டின் அளவீடுகளை மாற்றி ஆபத்து எப்படி மாறுகிறது என்று பாருங்கள்.",
        "whatif_cow": "தொடங்கும் மாடு",
        "reset": "அசல் ஆபத்து", "new": "சிமுலேட் ஆபத்து",
        "perf": "டெமோ தரவில் மாதிரி செயல்திறன்",
        "perf_note": ("மாதிரி இதே உருவகப்படுத்தப்பட்ட தரவில் பயிற்சி பெற்றது; எனவே இங்குள்ள எண்கள் அதிகமாகத் தெரியும். "
                      "திட்ட விளக்கக்காட்சியில் ROC-AUC 0.96, recall 75%, precision 0.33, சுமார் 4 நாள் முன்னெச்சரிக்கை."),
        "reported": "திட்ட அறிக்கையில் (சோதனைத் தரவு)",
        "this_data": "டெமோ தரவில் கணக்கிடப்பட்டது",
        "lead": "சராசரி முன்னெச்சரிக்கை நாட்கள்",
        "importance": "முக்கிய காரணிகள்",
        "note": "இது உருவகப்படுத்தப்பட்ட தரவு. உண்மையான பண்ணை தரவில் மீண்டும் பயிற்சி அளிக்க வேண்டும்.",
        "about_h": "MastiWatch பற்றி",
    },
}

NICE = {
    "scc_ma3": "Somatic cell count (3-day avg)", "scc_ma7": "Somatic cell count (7-day avg)",
    "scc_log": "Somatic cell count (log)", "scc_dev": "SCC vs cow's own normal",
    "scc_chg3": "SCC change (3 days)", "conductivity": "Milk conductivity",
    "conductivity_ma3": "Conductivity (3-day avg)", "conductivity_ma7": "Conductivity (7-day avg)",
    "conductivity_dev": "Conductivity vs cow's normal", "conductivity_chg3": "Conductivity change (3 days)",
    "yield_l": "Milk yield", "yield_l_ma3": "Yield (3-day avg)", "yield_l_ma7": "Yield (7-day avg)",
    "yield_l_dev": "Yield vs cow's normal", "yield_l_chg3": "Yield change (3 days)",
    "body_temp": "Body temperature", "body_temp_ma3": "Body temp (3-day avg)",
    "body_temp_ma7": "Body temp (7-day avg)", "body_temp_dev": "Body temp vs normal",
    "body_temp_chg3": "Body temp change", "hand_milking": "Hand milking", "hygiene": "Hygiene score",
    "parity": "Number of calvings", "humidity": "Humidity", "thi": "Heat stress (THI)",
    "thi_ma3": "Heat stress (3-day avg)", "temp": "Air temperature", "dim": "Days in milk",
    "breed_code": "Breed",
}

# ------------------------------------------------------------------ data / model
@st.cache_resource
def load_model():
    bundle = joblib.load(BASE / "mastitis_model.pkl")
    return bundle["model"], bundle["features"]


@st.cache_data
def load_scored_data():
    """Read the CSV once and score every row with the model."""
    model, feats = load_model()
    df = pd.read_csv(BASE / "mastitis_features.csv")
    df["risk"] = model.predict_proba(df[feats])[:, 1]
    return df


def band(p, high, med):
    return "HIGH" if p >= high else "MEDIUM" if p >= med else "LOW"


def contributions(model, feats, row_df):
    """Signed per-feature contributions (log-odds) for one row, via XGBoost."""
    try:
        import xgboost as xgb
        d = xgb.DMatrix(row_df[feats])
        c = model.get_booster().predict(d, pred_contribs=True)[0][:-1]
        return pd.Series(c, index=feats)
    except Exception:
        return None


def lead_time(df, thr):
    """For cows that became sick: days between first alert (inside the 5-day window) and onset."""
    out = []
    for _, g in df[df["onset_day"] >= 0].groupby("cow_id"):
        onset = g["onset_day"].iloc[0]
        pre = g[(g["day"] < onset) & (g["day"] >= onset - 5) & (g["risk"] >= thr)]
        if len(pre):
            out.append(onset - pre["day"].min())
    return float(np.mean(out)) if out else float("nan"), len(out)


# ------------------------------------------------------------------ chart helpers
def gauge(p, high, med):
    return go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": " %"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, med * 100], "color": "#c8e6c9"},
                         {"range": [med * 100, high * 100], "color": "#ffe0b2"},
                         {"range": [high * 100, 100], "color": "#ffcdd2"}]},
    )).update_layout(height=230, margin=dict(l=10, r=10, t=10, b=0))


def line(df, y, title, colour=BLUE, hlines=None, day=None, yfmt=None):
    f = go.Figure(go.Scatter(x=df["day"], y=df[y], mode="lines+markers", line=dict(color=colour)))
    for val, col, txt in (hlines or []):
        f.add_hline(y=val, line_dash="dash", line_color=col, annotation_text=txt)
    if day is not None:
        f.add_vline(x=day, line_dash="dot", line_color="grey")
    f.update_layout(title=title, height=270, margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_title="Day", yaxis_tickformat=yfmt)
    return f


# ================================================================== APP
lang = st.sidebar.radio("Language / மொழி", list(TEXT))
T = TEXT[lang]

model, FEATURES = load_model()
DF = load_scored_data()

st.sidebar.header(T["settings"])
days = sorted(DF["day"].unique())
day = st.sidebar.select_slider(T["day"], options=days, value=days[len(days) // 2])
breeds = [T["all"]] + sorted(DF["breed"].unique())
breed_sel = st.sidebar.selectbox(T["breed"], breeds)
st.sidebar.markdown("**" + T["thr"] + "**")
HIGH = st.sidebar.slider(T["thr_high"], 0.30, 0.95, 0.60, 0.05)
MED = st.sidebar.slider(T["thr_med"], 0.05, float(HIGH) - 0.05, min(0.30, HIGH - 0.05), 0.05)
st.sidebar.info(T["note"])

DF["band"] = DF["risk"].apply(lambda p: band(p, HIGH, MED))
VIEW = DF if breed_sel == T["all"] else DF[DF["breed"] == breed_sel]

st.title(T["title"])
st.caption(T["sub"])

# latest reading per cow on or before the chosen day
latest = (VIEW[VIEW["day"] <= day].sort_values("day").groupby("cow_id").tail(1))
latest = latest[latest["day"] == day]          # only cows measured on that day
today = latest.sort_values("risk", ascending=False)

tab_herd, tab_cow, tab_what, tab_perf, tab_about = st.tabs(T["tabs"])

# ------------------------------------------------------------------ TAB 1: herd
with tab_herd:
    st.subheader(T["herd"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🐄 " + T["cows"], len(today))
    c2.metric("🔴 " + T["high"], int((today.band == "HIGH").sum()))
    c3.metric("🟠 " + T["med"], int((today.band == "MEDIUM").sum()))
    c4.metric("🟢 " + T["low"], int((today.band == "LOW").sum()))

    if today.empty:
        st.warning(T["no_data"])
    else:
        a, b = st.columns(2)
        counts = today["band"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0)
        pie = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.5,
                               marker_colors=[BAND_COLOUR[k] for k in counts.index]))
        pie.update_layout(title=T["dist"], height=300, margin=dict(l=10, r=10, t=40, b=10))
        a.plotly_chart(pie, **STRETCH)

        br = today.groupby("breed")["risk"].mean().mul(100).sort_values()
        bar = go.Figure(go.Bar(x=br.values, y=br.index, orientation="h", marker_color=BLUE))
        bar.update_layout(title=T["bybreed"], height=300, xaxis_title="%",
                          margin=dict(l=10, r=10, t=40, b=10))
        b.plotly_chart(bar, **STRETCH)

        st.markdown("**" + T["table"] + "**")
        n = st.slider(T["top_n"], 5, min(50, len(today)), min(15, len(today)))
        show = today[["cow_id", "breed", "risk", "band", "yield_l", "conductivity",
                      "scc", "body_temp"]].head(n).copy()
        show["band"] = show["band"].map(lambda k: f"{BAND_ICON[k]} {k}")
        show.columns = ["Cow", "Breed", "Risk", "Level", "Yield (L)", "Conductivity",
                        "SCC (cells/mL)", "Body temp (°C)"]
        st.dataframe(
            show.round(2), **STRETCH, hide_index=True,
            column_config={"Risk": st.column_config.ProgressColumn(
                "Risk", min_value=0.0, max_value=1.0, format="%.2f")})
        st.download_button(T["download"], today.drop(columns=["band"]).to_csv(index=False).encode(),
                           f"mastiwatch_alerts_day{day}.csv", "text/csv")

# ------------------------------------------------------------------ TAB 2: cow detail
with tab_cow:
    st.subheader(T["tabs"][1])
    cow_ids = sorted(VIEW["cow_id"].unique())
    default = int(today.iloc[0]["cow_id"]) if len(today) else cow_ids[0]
    cow = st.selectbox(T["pick"], cow_ids, index=cow_ids.index(default))
    cow_df = DF[DF["cow_id"] == cow].sort_values("day")
    now = cow_df[cow_df["day"] <= day].tail(1)

    if now.empty:
        st.warning(T["no_data"])
    else:
        r = now.iloc[0]
        p = float(r["risk"]); b = band(p, HIGH, MED)
        left, right = st.columns([1, 2])
        with left:
            st.caption(T["risk"])
            st.plotly_chart(gauge(p, HIGH, MED), **STRETCH)
            st.markdown(f"### {BAND_ICON[b]} {b}")
            (st.error if b == "HIGH" else st.warning if b == "MEDIUM" else st.success)(T["actions"][b])
            st.markdown("**" + T["profile"] + "**")
            st.write(f"Breed: **{r['breed']}**  |  Calvings: **{int(r['parity'])}**  |  "
                     f"Days in milk: **{int(r['dim'])}**")
            st.write(f"Hand milking: **{'Yes' if r['hand_milking'] else 'No'}**  |  "
                     f"Hygiene score: **{int(r['hygiene'])}/5**")
        with right:
            st.markdown("**" + T["why"] + "**")
            st.caption(T["why_help"])
            c = contributions(model, FEATURES, now)
            if c is not None:
                top = c.reindex(c.abs().sort_values(ascending=False).index).head(8)[::-1]
                fig = go.Figure(go.Bar(
                    x=top.values, y=[NICE.get(f, f) for f in top.index], orientation="h",
                    marker_color=[RED if v > 0 else GREEN for v in top.values]))
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="Effect on risk (log-odds)")
                st.plotly_chart(fig, **STRETCH)
            else:
                st.info("Explanation not available for this model type.")

        st.plotly_chart(line(cow_df, "risk", T["prob"], RED, day=day,
                             hlines=[(HIGH, RED, "HIGH"), (MED, ORANGE, "MEDIUM")], yfmt=".0%"),
                        **STRETCH)
        st.markdown("**" + T["trend"] + "**")
        t1, t2 = st.columns(2)
        t1.plotly_chart(line(cow_df, "yield_l", "Milk yield (L)", BLUE, day=day), **STRETCH)
        t2.plotly_chart(line(cow_df, "conductivity", "Milk conductivity (mS/cm)", ORANGE, day=day),
                        **STRETCH)
        t3, t4 = st.columns(2)
        t3.plotly_chart(line(cow_df, "scc", "Somatic cell count (cells/mL)", RED, day=day),
                        **STRETCH)
        t4.plotly_chart(line(cow_df, "body_temp", "Body temperature (°C)", GREEN, day=day),
                        **STRETCH)

# ------------------------------------------------------------------ TAB 3: what-if
with tab_what:
    st.subheader(T["whatif"])
    st.caption(T["whatif_help"])
    ids = sorted(VIEW["cow_id"].unique())
    wcow = st.selectbox(T["whatif_cow"], ids,
                        index=ids.index(int(today.iloc[0]["cow_id"])) if len(today) else 0, key="wcow")
    base_df = DF[(DF["cow_id"] == wcow) & (DF["day"] <= day)].sort_values("day").tail(1)
    if base_df.empty:
        st.warning(T["no_data"])
    else:
        base = base_df.iloc[0]
        s1, s2, s3 = st.columns(3)
        scc_log = s1.slider("SCC (log scale)", 8.0, 15.0, float(base["scc_log"]), 0.1)
        scc_dev = s1.slider("SCC vs cow's normal", -1.0, 10.0, float(np.clip(base["scc_dev"], -1, 10)), 0.1)
        cond = s2.slider("Conductivity (mS/cm)", 3.5, 7.5, float(base["conductivity"]), 0.05)
        cond_dev = s2.slider("Conductivity vs normal", -0.5, 0.8, float(np.clip(base["conductivity_dev"], -0.5, 0.8)), 0.01)
        y_dev = s3.slider("Yield vs normal (L)", -3.0, 3.0, float(np.clip(base["yield_l_dev"], -3, 3)), 0.1)
        btemp = s3.slider("Body temp (°C)", 37.5, 41.0, float(np.clip(base["body_temp"], 37.5, 41)), 0.1)
        h1, h2 = st.columns(2)
        hyg = h1.slider("Hygiene score", 1, 5, int(base["hygiene"]))
        hand = h2.radio("Hand milking", [0, 1], index=int(base["hand_milking"]),
                        format_func=lambda v: "Yes" if v else "No", horizontal=True)

        new = base_df.copy()
        new["scc_log"] = scc_log; new["scc_dev"] = scc_dev
        new["conductivity"] = cond; new["conductivity_dev"] = cond_dev
        new["yield_l_dev"] = y_dev; new["body_temp"] = btemp
        new["hygiene"] = hyg; new["hand_milking"] = hand
        new_p = float(model.predict_proba(new[FEATURES])[:, 1][0])
        old_p = float(base["risk"])
        m1, m2 = st.columns(2)
        m1.metric(T["reset"], f"{old_p * 100:.0f} %")
        m2.metric(T["new"], f"{new_p * 100:.0f} %", f"{(new_p - old_p) * 100:+.0f} pts", delta_color="inverse")
        nb = band(new_p, HIGH, MED)
        (st.error if nb == "HIGH" else st.warning if nb == "MEDIUM" else st.success)(
            f"{BAND_ICON[nb]} {nb}: {T['actions'][nb]}")

# ------------------------------------------------------------------ TAB 4: performance
with tab_perf:
    st.subheader(T["perf"])
    st.warning(T["perf_note"])
    y, s = DF["label"], DF["risk"]
    pred = (s >= MED).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rec = tp / (tp + fn) if tp + fn else 0
    prec = tp / (tp + fp) if tp + fp else 0
    lt, n_cows = lead_time(DF, MED)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**" + T["reported"] + "**")
        k = st.columns(4)
        k[0].metric("ROC-AUC", "0.96"); k[1].metric("Recall", "75%")
        k[2].metric("Precision", "0.33"); k[3].metric("Early warning", "~4 days")
    with r2:
        st.markdown("**" + T["this_data"] + f"** (threshold {MED:.2f})")
        k = st.columns(4)
        k[0].metric("ROC-AUC", f"{roc_auc_score(y, s):.2f}"); k[1].metric("Recall", f"{rec:.0%}")
        k[2].metric("Precision", f"{prec:.2f}"); k[3].metric(T["lead"], f"{lt:.1f}")

    a, b = st.columns(2)
    fpr, tpr, _ = roc_curve(y, s)
    roc = go.Figure(go.Scatter(x=fpr, y=tpr, name="Model", line=dict(color=BLUE)))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="grey")))
    roc.update_layout(title="ROC curve", height=330, xaxis_title="False positive rate",
                      yaxis_title="True positive rate", margin=dict(l=10, r=10, t=40, b=10))
    a.plotly_chart(roc, **STRETCH)
    pr, rc, _ = precision_recall_curve(y, s)
    prf = go.Figure(go.Scatter(x=rc, y=pr, line=dict(color=ORANGE)))
    prf.update_layout(title="Precision-Recall curve", height=330, xaxis_title="Recall",
                      yaxis_title="Precision", margin=dict(l=10, r=10, t=40, b=10))
    b.plotly_chart(prf, **STRETCH)

    c, d = st.columns(2)
    cm = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Pred: healthy", "Pred: at risk"],
                              y=["Actual: healthy", "Actual: at risk"], colorscale="Blues",
                              text=[[tn, fp], [fn, tp]], texttemplate="%{text}", showscale=False))
    cm.update_layout(title="Confusion matrix", height=330, yaxis_autorange="reversed",
                     margin=dict(l=10, r=10, t=40, b=10))
    c.plotly_chart(cm, **STRETCH)
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values().tail(10)
    fi = go.Figure(go.Bar(x=imp.values, y=[NICE.get(f, f) for f in imp.index], orientation="h",
                          marker_color=BLUE))
    fi.update_layout(title=T["importance"], height=330, margin=dict(l=10, r=10, t=40, b=10))
    d.plotly_chart(fi, **STRETCH)

# ------------------------------------------------------------------ TAB 5: about
with tab_about:
    st.subheader(T["about_h"])
    st.markdown("""
**MastiWatch** is a software-only AI system that turns routinely recorded farm data
(milk yield, conductivity, somatic cell count, cow details, weather) into a **5-day
mastitis risk score** for every cow, ranked Low / Medium / High with a suggested action.

**Pipeline:** Farm data → engineered trend features (3/7-day averages, deviation from each cow's own
baseline, 3-day change) → XGBoost classifier → explanation of each alert → dashboard.

**Why it matters:** subclinical mastitis shows no visible signs yet quietly cuts milk yield and quality.
Today's tests (CMT, lab SCC) are reactive, manual and costly, and cannot forecast.

**Built for India:** heat stress (THI), breed, hand milking, hygiene score, English and Tamil.

**Tech stack:** Python, pandas, scikit-learn, XGBoost, Plotly, Streamlit.

**Team Med Sphere**: Asmiyanaseem S, Rakshita M, Sherifa Beevi N, Srirammuthiah C
PSNA College of Engineering and Technology.

**Limitation:** results come from simulated data and must be validated on real farm records
with veterinary colleges and cooperatives before deployment.
""")
    st.caption(T["note"])


