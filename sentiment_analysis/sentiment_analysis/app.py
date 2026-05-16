"""
app.py — Streamlit web interface for the Sentiment Analysis System
CSC 309 Mini Project #2  |  Okoro-Enyi Reginald  |  FUTO
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #1e1b4b; }
  [data-testid="stSidebar"] * { color: #e0e7ff !important; }
  .metric-card {
    background: #f8f7ff;
    border-left: 5px solid #7C3AED;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
  }
  .pos-badge { background:#dcfce7; color:#166534; padding:4px 14px;
               border-radius:999px; font-weight:700; font-size:1.1rem; }
  .neg-badge { background:#fee2e2; color:#991b1b; padding:4px 14px;
               border-radius:999px; font-weight:700; font-size:1.1rem; }
  .neu-badge { background:#fef9c3; color:#854d0e; padding:4px 14px;
               border-radius:999px; font-weight:700; font-size:1.1rem; }
  .stButton > button { background:#7C3AED; color:white; border:none;
                       border-radius:8px; padding:0.5rem 1.5rem; font-weight:600; }
  .stButton > button:hover { background:#6d28d9; }
</style>
""", unsafe_allow_html=True)

# ── Helper imports ────────────────────────────────────────────────────────────
from preprocessing.clean import clean_text
from baseline.vader_baseline import vader_predict, vader_scores

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MODEL_NAMES = {"lr": "Logistic Regression", "nb": "Naive Bayes", "svm": "Linear SVM"}
LABEL_MAP   = {0: "Negative", 1: "Positive", 2: "Neutral"}
BADGE_CLASS = {"Positive": "pos-badge", "Negative": "neg-badge", "Neutral": "neu-badge"}
EMOJI_MAP   = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(clf_name: str):
    path = os.path.join(OUTPUTS_DIR, f"model_{clf_name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data(show_spinner="Preparing dataset…")
def get_demo_data():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "demo_samples.csv")
    df = pd.read_csv(csv_path)
    label_map = {"Positive": 1, "Negative": 0, "Neutral": 2}
    df["label"] = df["text_label"].map(label_map)
    df["cleaned"] = df["text"].apply(clean_text)
    return df


@st.cache_data(show_spinner="Training model…")
def train_model(clf_name: str):
    from train import train
    df = get_demo_data()
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].rename(columns={"cleaned": "text"})[["text", "label"]]
    pipe = train(train_df, clf_name)
    return pipe


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Sentiment Analyser")
    st.markdown("*CSC 309 — AI Mini Project #2*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🔍 Live Prediction", "🏋️ Train & Evaluate", "📊 EDA & Word Clouds",
         "📖 About & Architecture"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    clf_choice = st.selectbox(
        "Active Model",
        options=list(MODEL_NAMES.keys()),
        format_func=lambda k: MODEL_NAMES[k],
    )
    st.markdown("---")
    st.caption("Okoro-Enyi Reginald · FUTO")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Live Prediction":
    st.title("🧠 Sentiment Analysis System")
    st.caption("CSC 309 Mini Project #2  |  NLP · TF-IDF · Logistic Regression")
    st.markdown("---")

    pipe = load_model(clf_choice)
    if pipe is None:
        st.warning(
            f"⚠️ No trained model found for **{MODEL_NAMES[clf_choice]}**. "
            "Go to **Train & Evaluate** to train one first."
        )

    # ── Input area ────────────────────────────────────────────────────────────
    st.subheader("Enter text to analyse")
    text_input = st.text_area(
        label="",
        height=160,
        placeholder="Type a movie review, tweet, product review, or any English sentence…",
        label_visibility="collapsed",
    )

    col_btn, col_ex = st.columns([1, 3])
    with col_btn:
        run = st.button("Analyse Sentiment", type="primary", use_container_width=True)
    with col_ex:
        example = st.selectbox(
            "Or try an example",
            ["",
             "This movie was absolutely brilliant and inspiring!",
             "The acting was terrible and the plot made no sense.",
             "It was okay, neither great nor particularly bad.",
             "I would not recommend this product to anyone.",
             "Hands down the best film I have seen this year."],
            label_visibility="collapsed",
        )
        if example and not text_input:
            text_input = example

    # ── Analysis ──────────────────────────────────────────────────────────────
    if (run or example) and text_input.strip():
        cleaned = clean_text(text_input)
        vader_label = vader_predict(text_input)
        vader_raw = vader_scores(text_input)

        st.markdown("---")
        c1, c2 = st.columns(2)

        # ML Model
        with c1:
            st.subheader("🤖 ML Model")
            if pipe is not None:
                pred_int = pipe.predict([cleaned])[0]
                ml_label = LABEL_MAP[pred_int]

                # Confidence
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba([cleaned])[0]
                    conf = proba[pred_int]
                    conf_display = True
                else:
                    conf = 0.0
                    conf_display = False

                badge = BADGE_CLASS[ml_label]
                emoji = EMOJI_MAP[ml_label]
                st.markdown(
                    f'<span class="{badge}">{emoji} {ml_label}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                if conf_display:
                    st.progress(float(conf), text=f"Confidence: {conf:.1%}")

                    # Bar chart of class probabilities
                    labels = [LABEL_MAP[i] for i in range(len(proba))]
                    colours = ["#fee2e2", "#dcfce7", "#fef9c3"][:len(proba)]
                    fig = go.Figure(go.Bar(
                        x=labels, y=[p * 100 for p in proba],
                        marker_color=colours,
                        text=[f"{p:.1%}" for p in proba],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        yaxis_title="Probability (%)",
                        yaxis_range=[0, 110],
                        showlegend=False,
                        height=280,
                        margin=dict(t=20, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Train a model first.")

        # VADER
        with c2:
            st.subheader("📖 VADER (Rule-based)")
            badge_v = BADGE_CLASS[vader_label]
            emoji_v = EMOJI_MAP[vader_label]
            st.markdown(
                f'<span class="{badge_v}">{emoji_v} {vader_label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.caption("VADER uses a sentiment lexicon — no training required")

            # Gauge chart for compound score
            compound = vader_raw["compound"]
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=compound,
                number={"suffix": "", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [-1, 1], "tickwidth": 1},
                    "bar": {"color": "#7C3AED", "thickness": 0.3},
                    "steps": [
                        {"range": [-1, -0.05], "color": "#fee2e2"},
                        {"range": [-0.05, 0.05], "color": "#fef9c3"},
                        {"range": [0.05, 1],    "color": "#dcfce7"},
                    ],
                    "threshold": {"line": {"color": "#7C3AED", "width": 3},
                                  "thickness": 0.75, "value": compound},
                },
                title={"text": "Compound Score"},
            ))
            fig2.update_layout(height=240, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig2, use_container_width=True)

            # Breakdown
            st.markdown("**Score breakdown**")
            dcols = st.columns(4)
            for (lbl, key), col in zip(
                [("Pos", "pos"), ("Neg", "neg"), ("Neu", "neu"), ("Compound", "compound")],
                dcols,
            ):
                col.metric(lbl, f"{vader_raw[key]:.3f}")

        # Preprocessed text
        with st.expander("🔎 View preprocessed text"):
            st.code(cleaned if cleaned else "(empty after cleaning)", language=None)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏋️ Train & Evaluate":
    st.title("🏋️ Train & Evaluate")
    st.markdown("Train a TF-IDF + ML pipeline on the demo dataset and view metrics.")
    st.markdown("---")

    df = get_demo_data()
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].rename(columns={"cleaned": "text"})[["text", "label"]]
    test_df  = df.iloc[split:].rename(columns={"cleaned": "text"})[["text", "label"]]

    st.info(
        f"Dataset: **{len(df)} samples** — "
        f"Train: {len(train_df)} | Test: {len(test_df)}"
    )

    col_train, col_cfg = st.columns([1, 2])
    with col_cfg:
        clf_to_train = st.selectbox(
            "Choose classifier",
            options=list(MODEL_NAMES.keys()),
            format_func=lambda k: MODEL_NAMES[k],
        )
    with col_train:
        do_train = st.button("🚀 Train Model", use_container_width=True)

    if do_train:
        with st.spinner(f"Training {MODEL_NAMES[clf_to_train]}…"):
            pipe = train_model(clf_to_train)
            train_model.clear()          # bust cache so re-train works
            pipe = train_model(clf_to_train)
        st.success(f"✅ {MODEL_NAMES[clf_to_train]} trained and saved.")

    # Evaluate
    pipe = load_model(clf_choice)
    if pipe is not None:
        st.markdown("---")
        st.subheader(f"📊 Evaluation — {MODEL_NAMES[clf_choice]}")
        from evaluate import evaluate, generate_wordclouds

        with st.spinner("Evaluating…"):
            results = evaluate(pipe, test_df)

        st.metric("Test Accuracy", f"{results['accuracy']:.2%}")

        tab_cm, tab_roc, tab_rep = st.tabs(["Confusion Matrix", "ROC Curve", "Classification Report"])

        with tab_cm:
            if os.path.exists(results["cm_path"]):
                st.image(results["cm_path"], width=500)

        with tab_roc:
            if results["roc_path"] and os.path.exists(results["roc_path"]):
                st.image(results["roc_path"], width=500)
            else:
                st.info("ROC curve available for binary Logistic Regression only.")

        with tab_rep:
            st.code(results["report"], language=None)

        # Word clouds
        st.markdown("---")
        st.subheader("☁️ Word Clouds")
        with st.spinner("Generating word clouds…"):
            pos_path, neg_path = generate_wordclouds(
                train_df.rename(columns={"text": "text"})
            )
        wc1, wc2 = st.columns(2)
        if pos_path and os.path.exists(pos_path):
            wc1.image(pos_path, caption="Positive Reviews", use_container_width=True)
        if neg_path and os.path.exists(neg_path):
            wc2.image(neg_path, caption="Negative Reviews", use_container_width=True)
    else:
        st.info(f"No model found for **{MODEL_NAMES[clf_choice]}**. Train one above.")

    # Classifier comparison table
    st.markdown("---")
    st.subheader("📋 Classifier Benchmark (Expected)")
    bench = pd.DataFrame([
        {"Model": "VADER (zero-shot)",      "Expected Accuracy": "68–72%",  "Notes": "Rule-based; no training"},
        {"Model": "Multinomial Naive Bayes","Expected Accuracy": "83–85%",  "Notes": "Fast; independence assumption"},
        {"Model": "Logistic Regression ★",  "Expected Accuracy": "87–89%",  "Notes": "Primary; calibrated probabilities"},
        {"Model": "Linear SVM",             "Expected Accuracy": "88–90%",  "Notes": "Highest accuracy; no proba"},
        {"Model": "BERT fine-tuned (ref.)", "Expected Accuracy": "93–95%",  "Notes": "GPU required; out of scope"},
    ])
    st.dataframe(bench, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Word Clouds":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    df = get_demo_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Positive", int((df["text_label"] == "Positive").sum()))
    col3.metric("Negative", int((df["text_label"] == "Negative").sum()))

    # Class distribution
    st.subheader("Class Distribution")
    counts = df["text_label"].value_counts()
    fig_pie = px.pie(
        names=counts.index, values=counts.values,
        color=counts.index,
        color_discrete_map={"Positive": "#86efac", "Negative": "#fca5a5", "Neutral": "#fde68a"},
        hole=0.45,
    )
    fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

    # Text length distribution
    st.subheader("Text Length Distribution")
    df["word_count"] = df["text"].str.split().str.len()
    fig_hist = px.histogram(
        df, x="word_count", color="text_label",
        nbins=30, barmode="overlay", opacity=0.7,
        color_discrete_map={"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#facc15"},
        labels={"word_count": "Word Count", "text_label": "Sentiment"},
    )
    fig_hist.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    # Sample rows
    st.subheader("Sample Data")
    st.dataframe(
        df[["text", "text_label", "word_count"]].sample(10, random_state=7)
            .rename(columns={"text": "Raw Text", "text_label": "Label",
                             "word_count": "Words"}),
        use_container_width=True, hide_index=True,
    )

    # VADER batch on demo
    st.subheader("VADER vs True Labels (Demo Set)")
    df["vader"] = df["text"].apply(vader_predict)
    match = (df["vader"] == df["text_label"]).mean()
    st.metric("VADER Agreement with Labels", f"{match:.1%}")

    fig_vader = px.histogram(
        df, x="vader", color="text_label",
        barmode="group",
        color_discrete_map={"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#facc15"},
        labels={"vader": "VADER Prediction", "text_label": "True Label"},
    )
    fig_vader.update_layout(height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig_vader, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT & ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 About & Architecture":
    st.title("📖 About This System")
    st.markdown("---")

    st.markdown("""
**CSC 309 — Artificial Intelligence Mini Project #2**

| Field | Detail |
|---|---|
| **Student** | Okoro-Enyi Reginald |
| **Institution** | Federal University of Technology Owerri |
| **Course** | CSC 309 — Artificial Intelligence |
| **Version** | 1.0 — Initial Submission |
| **Date** | 11 May 2026 |
""")

    st.markdown("---")
    st.subheader("🔄 NLP Pipeline")
    st.markdown("""
```
INPUT TEXT (raw review / tweet / sentence)
        ↓
┌─────────────────────────────┐
│   TEXT PREPROCESSING        │
│   Lowercase → Strip HTML    │
│   Remove punctuation        │
│   Tokenise (NLTK)           │
│   Remove stop-words         │
│   Lemmatisation             │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   FEATURE EXTRACTION        │
│   TF-IDF (max 50k terms)    │
│   Unigrams + Bigrams        │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   CLASSIFIER                │
│   Naive Bayes / LR / SVM    │
└─────────────────────────────┘
        ↓
  SENTIMENT LABEL:
  POSITIVE / NEGATIVE / NEUTRAL
        ↓
  STREAMLIT WEB INTERFACE
```
""")

    st.subheader("📦 Key Design Decisions")
    decisions = pd.DataFrame([
        {"Decision": "TF-IDF over BoW",          "Rationale": "Penalises common words; improves discriminative power"},
        {"Decision": "Bigrams (1,2)",             "Rationale": "Captures negation: 'not good' vs 'good'"},
        {"Decision": "Logistic Regression (primary)", "Rationale": "Best accuracy/probability balance for Streamlit display"},
        {"Decision": "Sklearn Pipeline",          "Rationale": "Chains vectoriser + classifier; prevents data leakage"},
        {"Decision": "VADER as baseline",         "Rationale": "Zero-shot comparison; needs no labelled data"},
    ])
    st.dataframe(decisions, use_container_width=True, hide_index=True)

    st.subheader("🔮 Future Work")
    st.markdown("""
- Fine-tune **DistilBERT / RoBERTa** for transformer-level accuracy (93%+)
- **Aspect-based sentiment** — identify which parts of a review are positive/negative
- **Multilingual** support — XLM-RoBERTa for Igbo, Yoruba, Hausa
- **LIME / SHAP** explainability to highlight influential words
- **FastAPI + React** dashboard for multi-user production deployment
- **Real-time Twitter/X** streaming for live public sentiment monitoring
""")

    st.subheader("📚 Dependencies")
    deps = pd.DataFrame([
        {"Package": "scikit-learn ≥1.4", "Purpose": "TF-IDF, classifiers, evaluation"},
        {"Package": "nltk ≥3.8",         "Purpose": "Tokenisation, stop-words, VADER"},
        {"Package": "pandas ≥2.1",       "Purpose": "Dataset management"},
        {"Package": "streamlit ≥1.35",   "Purpose": "Interactive web interface"},
        {"Package": "plotly ≥5.18",      "Purpose": "Interactive charts"},
        {"Package": "wordcloud ≥1.9",    "Purpose": "Word cloud visualisation"},
        {"Package": "joblib ≥1.3",       "Purpose": "Model serialisation (.pkl)"},
    ])
    st.dataframe(deps, use_container_width=True, hide_index=True)
