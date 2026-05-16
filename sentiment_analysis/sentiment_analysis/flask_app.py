"""
flask_app.py — Flask web interface for the Sentiment Analysis System
CSC 309 Mini Project #2  |  Okoro-Enyi Reginald  |  FUTO
Run: python flask_app.py
"""
import sys, os, json, re, io, base64
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flask import Flask, render_template_string, request, jsonify
from preprocessing.clean import clean_text
from baseline.vader_baseline import vader_predict, vader_scores

app = Flask(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

LABEL_MAP   = {0: "Negative", 1: "Positive", 2: "Neutral"}
MODEL_NAMES = {"lr": "Logistic Regression", "nb": "Naive Bayes", "svm": "Linear SVM"}


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_model(clf_name="lr"):
    path = os.path.join(OUTPUTS_DIR, f"model_{clf_name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_proba_chart(labels, probas):
    colours = ["#f87171", "#4ade80", "#fbbf24"][:len(labels)]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    bars = ax.bar(labels, [p * 100 for p in probas], color=colours,
                  edgecolor="white", linewidth=1.5, zorder=3)
    for bar, p in zip(bars, probas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{p:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Probability (%)", fontsize=9)
    ax.set_facecolor("#f8f7ff")
    fig.patch.set_facecolor("#f8f7ff")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def build_compound_gauge(compound):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    theta = np.linspace(np.pi, 0, 200)
    # Background arc
    ax.plot(theta, [1] * 200, lw=20, color="#e5e7eb", solid_capstyle="round",
            transform=ax.transData, clip_on=False)
    # Coloured sections
    for t, c in [(np.linspace(np.pi, np.pi * 0.6, 50), "#fca5a5"),
                 (np.linspace(np.pi * 0.6, np.pi * 0.4, 30), "#fde68a"),
                 (np.linspace(np.pi * 0.4, 0, 50), "#86efac")]:
        ax.plot(t, [1] * len(t), lw=20, color=c, solid_capstyle="butt",
                clip_on=False)
    # Needle
    needle_angle = np.pi - np.pi * ((compound + 1) / 2)
    ax.annotate("", xy=(needle_angle, 0.9), xytext=(needle_angle, 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2))
    ax.set_ylim(0, 1.3)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines["polar"].set_visible(False)
    ax.text(np.pi / 2, -0.35, f"Compound: {compound:+.3f}", ha="center",
            va="top", fontsize=11, fontweight="bold", color="#7c3aed",
            transform=ax.transData)
    fig.patch.set_facecolor("#f8f7ff")
    ax.set_facecolor("#f8f7ff")
    plt.tight_layout(pad=0)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def build_confusion_matrix_chart(pipe, test_df):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_pred = pipe.predict(test_df["cleaned"])
    y_true = test_df["label"]
    n = len(np.unique(y_true))
    labels = ["Negative", "Positive", "Neutral"][:n]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Purples", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold", pad=10)
    fig.patch.set_facecolor("#f8f7ff")
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def build_distribution_chart(df):
    counts = df["text_label"].value_counts()
    colours = {"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#fbbf24"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Pie
    wedge_colors = [colours.get(l, "#ccc") for l in counts.index]
    axes[0].pie(counts.values, labels=counts.index, colors=wedge_colors,
                autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white", "lw": 2})
    axes[0].set_title("Class Distribution", fontweight="bold")

    # Word count histogram
    df_copy = df.copy()
    df_copy["word_count"] = df_copy["text"].str.split().str.len()
    for lbl in ["Positive", "Negative", "Neutral"]:
        subset = df_copy[df_copy["text_label"] == lbl]["word_count"]
        if len(subset):
            axes[1].hist(subset.values, bins=20, alpha=0.6, color=colours[lbl],
                         label=lbl, edgecolor="white")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Text Length Distribution", fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.patch.set_facecolor("#f8f7ff")
    for ax in axes:
        ax.set_facecolor("#f8f7ff")
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def get_demo_df():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "demo_samples.csv")
    df = pd.read_csv(csv_path)
    label_map = {"Positive": 1, "Negative": 0, "Neutral": 2}
    df["label"] = df["text_label"].map(label_map)
    df["cleaned"] = df["text"].apply(clean_text)
    return df


# ── HTML template ────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment Analysis System — CSC 309</title>
<style>
  :root {
    --purple:#7c3aed; --purple-light:#ede9fe; --purple-dark:#5b21b6;
    --green:#dcfce7; --green-text:#166534;
    --red:#fee2e2; --red-text:#991b1b;
    --yellow:#fef9c3; --yellow-text:#854d0e;
    --bg:#f5f3ff; --card:#ffffff; --border:#e5e7eb;
    --sidebar:#1e1b4b; --sidebar-text:#e0e7ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: var(--bg); color: #1f2937; }
  .layout { display: flex; min-height: 100vh; }

  /* Sidebar */
  .sidebar {
    width: 240px; min-width: 240px; background: var(--sidebar);
    padding: 1.5rem 1rem; display: flex; flex-direction: column; gap: 0.5rem;
  }
  .sidebar h2 { color: #e0e7ff; font-size: 1.1rem; margin-bottom: 0.25rem; }
  .sidebar .caption { color: #a5b4fc; font-size: 0.75rem; margin-bottom: 1rem; }
  .sidebar hr { border-color: #312e81; margin: 0.5rem 0; }
  .nav-btn {
    display: block; width: 100%; text-align: left;
    background: transparent; border: none; color: #c7d2fe;
    padding: 0.5rem 0.75rem; border-radius: 6px; cursor: pointer;
    font-size: 0.88rem; transition: background 0.15s;
  }
  .nav-btn:hover, .nav-btn.active { background: #312e81; color: #e0e7ff; }
  .sidebar select {
    width: 100%; padding: 0.4rem; border-radius: 6px; border: 1px solid #4338ca;
    background: #312e81; color: #e0e7ff; font-size: 0.85rem; margin-top: 0.25rem;
  }
  .sidebar label { color: #a5b4fc; font-size: 0.78rem; display: block; margin-top: 0.5rem; }

  /* Main */
  .main { flex: 1; padding: 2rem 2.5rem; max-width: 960px; }
  h1 { font-size: 1.8rem; color: var(--purple-dark); margin-bottom: 0.25rem; }
  h2 { font-size: 1.25rem; color: #374151; margin: 1.25rem 0 0.5rem; }
  .caption-main { color: #6b7280; font-size: 0.85rem; margin-bottom: 1rem; }
  hr.divider { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }

  /* Cards */
  .card {
    background: var(--card); border-radius: 12px; padding: 1.25rem 1.5rem;
    border: 1px solid var(--border); box-shadow: 0 1px 4px rgba(0,0,0,.05);
    margin-bottom: 1.2rem;
  }
  .card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }

  /* Badge */
  .badge {
    display: inline-block; padding: 4px 16px; border-radius: 999px;
    font-weight: 700; font-size: 1.05rem; margin-bottom: 0.75rem;
  }
  .badge-pos { background: var(--green); color: var(--green-text); }
  .badge-neg { background: var(--red); color: var(--red-text); }
  .badge-neu { background: var(--yellow); color: var(--yellow-text); }

  /* Inputs */
  textarea {
    width: 100%; height: 140px; padding: 0.75rem; border-radius: 8px;
    border: 1px solid var(--border); font-size: 0.95rem; resize: vertical;
    font-family: inherit; background: #fafaf9;
  }
  textarea:focus { outline: 2px solid var(--purple); border-color: transparent; }
  select.input-select {
    padding: 0.5rem 0.75rem; border-radius: 8px; border: 1px solid var(--border);
    font-size: 0.9rem; background: white; min-width: 280px;
  }
  .btn {
    background: var(--purple); color: white; border: none; border-radius: 8px;
    padding: 0.55rem 1.5rem; font-size: 0.95rem; font-weight: 600;
    cursor: pointer; transition: background 0.15s;
  }
  .btn:hover { background: var(--purple-dark); }
  .btn-row { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-top: 0.75rem; }

  /* Progress bar */
  .progress-wrap { margin: 0.4rem 0 0.6rem; }
  .progress-label { font-size: 0.8rem; color: #6b7280; margin-bottom: 3px; }
  .progress-bar { height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
  .progress-fill { height: 100%; background: var(--purple); border-radius: 999px; transition: width 0.4s; }

  /* Metrics row */
  .metrics { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .metric-box {
    background: var(--purple-light); border-radius: 8px; padding: 0.6rem 1rem;
    min-width: 110px; text-align: center;
  }
  .metric-value { font-size: 1.4rem; font-weight: 700; color: var(--purple-dark); }
  .metric-label { font-size: 0.75rem; color: #6b7280; margin-top: 2px; }

  /* VADER scores row */
  .vader-scores { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 0.6rem; }
  .vader-score { background: #f3f4f6; border-radius: 6px; padding: 0.35rem 0.75rem; font-size: 0.85rem; }
  .vader-score span { font-weight: 700; color: var(--purple); }

  /* Table */
  table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
  th { background: var(--purple-light); color: var(--purple-dark); text-align: left;
       padding: 0.5rem 0.75rem; font-weight: 600; }
  td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #f3f4f6; }
  tr:hover td { background: #fafaf9; }

  /* Code block */
  pre { background: #1e1b4b; color: #c7d2fe; padding: 1rem; border-radius: 8px;
        font-size: 0.82rem; overflow-x: auto; line-height: 1.6; white-space: pre-wrap; }

  /* Alert */
  .alert { background: var(--yellow); border-left: 4px solid #f59e0b;
           padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.9rem; color: #92400e; }

  /* Loading */
  .loading { display: none; color: var(--purple); font-size: 0.9rem; }

  /* Section visibility */
  .section { display: none; }
  .section.active { display: block; }

  /* Responsive */
  @media (max-width: 640px) {
    .card-grid { grid-template-columns: 1fr; }
    .sidebar { width: 190px; min-width: 190px; }
    .main { padding: 1rem 1rem; }
  }
</style>
</head>
<body>
<div class="layout">

<!-- Sidebar -->
<nav class="sidebar">
  <h2>🧠 Sentiment<br>Analyser</h2>
  <p class="caption">CSC 309 – AI Mini Project #2</p>
  <hr>
  <button class="nav-btn active" onclick="showSection('predict', this)">🔍 Live Prediction</button>
  <button class="nav-btn" onclick="showSection('train', this)">🏋️ Train & Evaluate</button>
  <button class="nav-btn" onclick="showSection('eda', this)">📊 EDA & Charts</button>
  <button class="nav-btn" onclick="showSection('about', this)">📖 About</button>
  <hr>
  <label>Active Model</label>
  <select id="globalModel" onchange="updateModel()">
    <option value="lr">Logistic Regression</option>
    <option value="nb">Naive Bayes</option>
    <option value="svm">Linear SVM</option>
  </select>
  <hr>
  <p style="color:#6b7280;font-size:0.72rem;margin-top:auto">
    Okoro-Enyi Reginald<br>FUTO · May 2026
  </p>
</nav>

<!-- Main content -->
<main class="main">

<!-- ══ PREDICT ══ -->
<section id="sec-predict" class="section active">
  <h1>🧠 Sentiment Analysis System</h1>
  <p class="caption-main">CSC 309 Mini Project #2 &nbsp;|&nbsp; NLP · TF-IDF · Logistic Regression</p>
  <hr class="divider">

  <div class="card">
    <h2 style="margin-top:0">Enter text to analyse</h2>
    <textarea id="inputText" placeholder="Type a movie review, tweet, product review, or any English sentence…"></textarea>
    <div class="btn-row">
      <button class="btn" onclick="analyse()">Analyse Sentiment</button>
      <select class="input-select" id="exampleSelect" onchange="fillExample()">
        <option value="">— Try an example —</option>
        <option value="This movie was absolutely brilliant and inspiring!">✅ Positive: Brilliant movie</option>
        <option value="The acting was terrible and the plot made no sense.">❌ Negative: Terrible film</option>
        <option value="It was okay, neither great nor particularly bad.">😐 Neutral: Just okay</option>
        <option value="I would not recommend this product to anyone.">❌ Negative: Would not recommend</option>
        <option value="Hands down the best film I have seen this year.">✅ Positive: Best film</option>
        <option value="The product broke after two days of use.">❌ Negative: Broke quickly</option>
        <option value="It does the job but nothing to get excited about.">😐 Neutral: Does the job</option>
      </select>
    </div>
    <p class="loading" id="loadingPredict">⏳ Analysing…</p>
  </div>

  <div id="results" style="display:none">
    <div class="card-grid">
      <!-- ML Model -->
      <div class="card">
        <h2 style="margin-top:0">🤖 ML Model</h2>
        <div id="mlBadge"></div>
        <div class="progress-wrap" id="confWrap">
          <div class="progress-label" id="confLabel"></div>
          <div class="progress-bar"><div class="progress-fill" id="confBar" style="width:0%"></div></div>
        </div>
        <img id="probaChart" style="width:100%;border-radius:8px;margin-top:0.5rem" alt="Probability chart">
      </div>
      <!-- VADER -->
      <div class="card">
        <h2 style="margin-top:0">📖 VADER (Rule-based)</h2>
        <div id="vaderBadge"></div>
        <p style="font-size:0.82rem;color:#6b7280;margin-bottom:0.5rem">VADER uses a sentiment lexicon — no training required</p>
        <img id="gaugeChart" style="width:100%;border-radius:8px" alt="VADER gauge">
        <div class="vader-scores">
          <div class="vader-score">Pos: <span id="vs-pos"></span></div>
          <div class="vader-score">Neg: <span id="vs-neg"></span></div>
          <div class="vader-score">Neu: <span id="vs-neu"></span></div>
          <div class="vader-score">Compound: <span id="vs-comp"></span></div>
        </div>
      </div>
    </div>
    <details style="margin-top:-0.5rem">
      <summary style="cursor:pointer;color:var(--purple);font-size:0.88rem;padding:0.4rem 0">🔎 View preprocessed text</summary>
      <pre id="cleanedText" style="margin-top:0.5rem"></pre>
    </details>
  </div>
</section>

<!-- ══ TRAIN ══ -->
<section id="sec-train" class="section">
  <h1>🏋️ Train & Evaluate</h1>
  <p class="caption-main">Train a TF-IDF + ML pipeline on the demo dataset and view evaluation metrics.</p>
  <hr class="divider">

  <div class="card">
    <div class="btn-row">
      <select class="input-select" id="trainModel">
        <option value="lr">Logistic Regression</option>
        <option value="nb">Naive Bayes</option>
        <option value="svm">Linear SVM</option>
      </select>
      <button class="btn" onclick="trainModel()">🚀 Train Model</button>
      <span id="trainStatus" style="font-size:0.88rem;color:#6b7280"></span>
    </div>
  </div>

  <div id="evalResults" style="display:none">
    <div class="metrics" id="metricsRow"></div>
    <div class="card-grid">
      <div class="card">
        <h2 style="margin-top:0">Confusion Matrix</h2>
        <img id="cmChart" style="width:100%;border-radius:8px" alt="Confusion matrix">
      </div>
      <div class="card">
        <h2 style="margin-top:0">Classification Report</h2>
        <pre id="reportPre" style="font-size:0.78rem"></pre>
      </div>
    </div>
    <div class="card">
      <h2 style="margin-top:0">☁️ Word Clouds</h2>
      <div class="card-grid" id="wordcloudSection">
        <div><img id="wcPos" style="width:100%;border-radius:8px" alt="Positive word cloud"><p style="text-align:center;font-size:0.82rem;color:#6b7280">Positive Reviews</p></div>
        <div><img id="wcNeg" style="width:100%;border-radius:8px" alt="Negative word cloud"><p style="text-align:center;font-size:0.82rem;color:#6b7280">Negative Reviews</p></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0">📋 Classifier Benchmark (Expected)</h2>
    <table>
      <thead><tr><th>Model</th><th>Expected Accuracy</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td>VADER (zero-shot)</td><td>68–72%</td><td>Rule-based; no training</td></tr>
        <tr><td>Multinomial Naive Bayes</td><td>83–85%</td><td>Fast; independence assumption</td></tr>
        <tr style="background:#ede9fe"><td><strong>Logistic Regression ★</strong></td><td><strong>87–89%</strong></td><td>Primary model; calibrated probabilities</td></tr>
        <tr><td>Linear SVM</td><td>88–90%</td><td>Highest accuracy; no probabilities</td></tr>
        <tr><td>BERT fine-tuned (ref.)</td><td>93–95%</td><td>GPU required; out of scope</td></tr>
      </tbody>
    </table>
  </div>
</section>

<!-- ══ EDA ══ -->
<section id="sec-eda" class="section">
  <h1>📊 Exploratory Data Analysis</h1>
  <hr class="divider">
  <div class="metrics" id="edaMetrics"></div>
  <div class="card">
    <h2 style="margin-top:0">Distribution & Text Length</h2>
    <img id="distChart" style="width:100%;border-radius:8px" alt="Distribution chart">
  </div>
  <div class="card">
    <h2 style="margin-top:0">VADER vs True Labels</h2>
    <div class="metrics" id="vaderMetrics"></div>
    <img id="vaderChart" style="width:100%;border-radius:8px" alt="VADER chart">
  </div>
  <div class="card">
    <h2 style="margin-top:0">Sample Data</h2>
    <div id="sampleTable"></div>
  </div>
</section>

<!-- ══ ABOUT ══ -->
<section id="sec-about" class="section">
  <h1>📖 About This System</h1>
  <hr class="divider">
  <div class="card">
    <table>
      <tr><th>Field</th><th>Detail</th></tr>
      <tr><td>Student</td><td>Okoro-Enyi Reginald</td></tr>
      <tr><td>Institution</td><td>Federal University of Technology Owerri</td></tr>
      <tr><td>Course</td><td>CSC 309 — Artificial Intelligence</td></tr>
      <tr><td>Project</td><td>Project 2 of 24 — Sentiment Analysis System</td></tr>
      <tr><td>Version</td><td>1.0 — Initial Submission</td></tr>
      <tr><td>Date</td><td>11 May 2026</td></tr>
    </table>
  </div>
  <div class="card">
    <h2 style="margin-top:0">🔄 NLP Pipeline</h2>
    <pre>
INPUT TEXT (raw review / tweet / sentence)
      ↓
┌─────────────────────────────┐
│  TEXT PREPROCESSING         │
│  Lowercase → Strip HTML     │
│  Remove punctuation         │
│  Tokenise → Stop-word removal│
│  Lemmatisation              │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│  FEATURE EXTRACTION         │
│  TF-IDF (max 50k terms)     │
│  Unigrams + Bigrams (1,2)   │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│  CLASSIFIER                 │
│  Naive Bayes / LR / SVM     │
└─────────────────────────────┘
      ↓
SENTIMENT LABEL: POSITIVE / NEGATIVE / NEUTRAL
      ↓
WEB INTERFACE (Flask / Streamlit)
</pre>
  </div>
  <div class="card">
    <h2 style="margin-top:0">📦 Key Design Decisions</h2>
    <table>
      <tr><th>Decision</th><th>Rationale</th></tr>
      <tr><td>TF-IDF over BoW</td><td>Penalises common words; improves discriminative power with no extra cost</td></tr>
      <tr><td>Bigrams (1,2)</td><td>Captures negation context: 'not good' vs 'good'</td></tr>
      <tr><td>Logistic Regression (primary)</td><td>Best accuracy/probability balance for confidence display</td></tr>
      <tr><td>Sklearn Pipeline</td><td>Chains vectoriser + classifier; prevents data leakage</td></tr>
      <tr><td>VADER as baseline</td><td>Zero-shot comparison; needs no labelled data</td></tr>
    </table>
  </div>
  <div class="card">
    <h2 style="margin-top:0">📚 Dependencies</h2>
    <table>
      <tr><th>Package</th><th>Purpose</th></tr>
      <tr><td>scikit-learn ≥1.4</td><td>TF-IDF, classifiers, evaluation</td></tr>
      <tr><td>nltk ≥3.8 (optional)</td><td>Tokenisation, stop-words, VADER (falls back to pure Python)</td></tr>
      <tr><td>pandas ≥2.1</td><td>Dataset management</td></tr>
      <tr><td>streamlit ≥1.35</td><td>Interactive web interface (alternative deployment)</td></tr>
      <tr><td>flask ≥3.0</td><td>Web framework (this deployment)</td></tr>
      <tr><td>matplotlib ≥3.8</td><td>Charts and word cloud visualisation</td></tr>
      <tr><td>joblib ≥1.3</td><td>Model serialisation (.pkl)</td></tr>
    </table>
  </div>
  <div class="card">
    <h2 style="margin-top:0">🔮 Future Work</h2>
    <ul style="padding-left:1.5rem;line-height:1.9;font-size:0.9rem">
      <li>Fine-tune <strong>DistilBERT / RoBERTa</strong> for 93%+ accuracy</li>
      <li><strong>Aspect-based sentiment</strong> — identify which parts are positive/negative</li>
      <li><strong>Multilingual</strong> support — XLM-RoBERTa for Igbo, Yoruba, Hausa</li>
      <li><strong>LIME / SHAP</strong> explainability — highlight influential words</li>
      <li><strong>FastAPI + React</strong> dashboard for multi-user production deployment</li>
      <li><strong>Real-time Twitter/X streaming</strong> for live public sentiment monitoring</li>
    </ul>
  </div>
</section>

</main>
</div>

<script>
let currentModel = 'lr';

function showSection(name, btn) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('sec-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'eda') loadEDA();
}

function updateModel() {
  currentModel = document.getElementById('globalModel').value;
}

function fillExample() {
  const val = document.getElementById('exampleSelect').value;
  if (val) { document.getElementById('inputText').value = val; analyse(); }
}

function badgeClass(label) {
  return label === 'Positive' ? 'pos' : label === 'Negative' ? 'neg' : 'neu';
}
function badgeEmoji(label) {
  return label === 'Positive' ? '🟢' : label === 'Negative' ? '🔴' : '🟡';
}

async function analyse() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) return;
  document.getElementById('loadingPredict').style.display = 'block';
  document.getElementById('results').style.display = 'none';

  const resp = await fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, model: currentModel})
  });
  const data = await resp.json();
  document.getElementById('loadingPredict').style.display = 'none';

  // ML
  const bc = badgeClass(data.ml_label);
  document.getElementById('mlBadge').innerHTML =
    `<span class="badge badge-${bc}">${badgeEmoji(data.ml_label)} ${data.ml_label}</span>`;
  if (data.confidence !== null) {
    document.getElementById('confLabel').textContent = `Confidence: ${(data.confidence*100).toFixed(1)}%`;
    document.getElementById('confBar').style.width = (data.confidence*100) + '%';
    document.getElementById('confWrap').style.display = 'block';
  } else {
    document.getElementById('confWrap').style.display = 'none';
  }
  document.getElementById('probaChart').src = 'data:image/png;base64,' + data.proba_chart;

  // VADER
  const vc = badgeClass(data.vader_label);
  document.getElementById('vaderBadge').innerHTML =
    `<span class="badge badge-${vc}">${badgeEmoji(data.vader_label)} ${data.vader_label}</span>`;
  document.getElementById('gaugeChart').src = 'data:image/png;base64,' + data.gauge_chart;
  document.getElementById('vs-pos').textContent = data.vader_scores.pos;
  document.getElementById('vs-neg').textContent = data.vader_scores.neg;
  document.getElementById('vs-neu').textContent = data.vader_scores.neu;
  document.getElementById('vs-comp').textContent = data.vader_scores.compound;
  document.getElementById('cleanedText').textContent = data.cleaned || '(empty after cleaning)';

  document.getElementById('results').style.display = 'block';
}

async function trainModel() {
  const name = document.getElementById('trainModel').value;
  document.getElementById('trainStatus').textContent = '⏳ Training…';
  const resp = await fetch('/api/train', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model: name})
  });
  const data = await resp.json();
  document.getElementById('trainStatus').textContent = data.message;
  if (data.accuracy) {
    document.getElementById('evalResults').style.display = 'block';
    document.getElementById('metricsRow').innerHTML =
      `<div class="metric-box"><div class="metric-value">${(data.accuracy*100).toFixed(1)}%</div><div class="metric-label">Test Accuracy</div></div>`;
    document.getElementById('reportPre').textContent = data.report;
    document.getElementById('cmChart').src = 'data:image/png;base64,' + data.cm_chart;
    if (data.wc_pos) document.getElementById('wcPos').src = 'data:image/png;base64,' + data.wc_pos;
    if (data.wc_neg) document.getElementById('wcNeg').src = 'data:image/png;base64,' + data.wc_neg;
  }
}

async function loadEDA() {
  if (document.getElementById('distChart').src) return;
  const resp = await fetch('/api/eda');
  const data = await resp.json();

  document.getElementById('edaMetrics').innerHTML = [
    ['Total Samples', data.total],
    ['Positive', data.positive],
    ['Negative', data.negative],
    ['Neutral', data.neutral],
  ].map(([l,v]) =>
    `<div class="metric-box"><div class="metric-value">${v}</div><div class="metric-label">${l}</div></div>`
  ).join('');

  document.getElementById('distChart').src = 'data:image/png;base64,' + data.dist_chart;
  document.getElementById('vaderChart').src = 'data:image/png;base64,' + data.vader_chart;
  document.getElementById('vaderMetrics').innerHTML =
    `<div class="metric-box"><div class="metric-value">${(data.vader_agreement*100).toFixed(1)}%</div><div class="metric-label">VADER Agreement</div></div>`;

  document.getElementById('sampleTable').innerHTML = `
    <table>
      <thead><tr><th>Text</th><th>Label</th><th>Words</th></tr></thead>
      <tbody>${data.samples.map(r =>
        `<tr><td>${r.text}</td><td>${r.label}</td><td>${r.words}</td></tr>`
      ).join('')}</tbody>
    </table>`;
}
</script>
</body>
</html>"""


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json()
    text = body.get("text", "")
    clf_name = body.get("model", "lr")

    cleaned = clean_text(text)
    vader_label = vader_predict(text)
    vscores = vader_scores(text)

    pipe = load_model(clf_name)
    ml_label, confidence, proba_vals, proba_labels = None, None, [], []

    if pipe:
        pred_int = int(pipe.predict([cleaned])[0])
        ml_label = LABEL_MAP[pred_int]
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba([cleaned])[0]
            confidence = float(proba[pred_int])
            proba_labels = [LABEL_MAP[i] for i in range(len(proba))]
            proba_vals = proba.tolist()
        else:
            confidence = None
    else:
        ml_label = "Model not trained"

    # Charts
    proba_chart = build_proba_chart(proba_labels or ["Neg", "Pos"], proba_vals or [0.5, 0.5])
    gauge_chart = build_compound_gauge(vscores["compound"])

    return jsonify({
        "ml_label": ml_label,
        "confidence": confidence,
        "proba_chart": proba_chart,
        "vader_label": vader_label,
        "vader_scores": vscores,
        "gauge_chart": gauge_chart,
        "cleaned": cleaned,
    })


@app.route("/api/train", methods=["POST"])
def train_route():
    from train import train
    from evaluate import evaluate, generate_wordclouds
    from sklearn.metrics import accuracy_score, classification_report
    import traceback

    body = request.get_json()
    clf_name = body.get("model", "lr")

    try:
        df = get_demo_df()
        split = int(len(df) * 0.8)
        train_df = df.iloc[:split]
        test_df  = df.iloc[split:]

        pipe = train(train_df.rename(columns={"cleaned": "text"}), clf_name)

        y_pred = pipe.predict(test_df["cleaned"])
        acc = accuracy_score(test_df["label"], y_pred)
        labels_list = [LABEL_MAP[i] for i in sorted(test_df["label"].unique())]
        report = classification_report(
            test_df["label"], y_pred, target_names=labels_list
        )
        cm_b64 = build_confusion_matrix_chart(pipe, test_df)

        # Word clouds (matplotlib only)
        wc_pos_b64 = wc_neg_b64 = None
        try:
            from wordcloud import WordCloud
            for label_int, label_name in [(1, "positive"), (0, "negative")]:
                subset = train_df[train_df["label"] == label_int]["text"]
                blob = " ".join(subset.tolist())
                wc = WordCloud(width=700, height=350, background_color="white",
                               max_words=100).generate(blob)
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                b64 = fig_to_b64(fig); plt.close(fig)
                if label_name == "positive": wc_pos_b64 = b64
                else: wc_neg_b64 = b64
        except ImportError:
            # wordcloud not installed — show frequency bar charts instead
            for label_int, label_name, color in [
                (1, "positive", "#4ade80"), (0, "negative", "#f87171")
            ]:
                from collections import Counter
                subset = " ".join(train_df[train_df["label"] == label_int]["text"].tolist())
                words = [w for w in subset.split() if len(w) > 3]
                top = Counter(words).most_common(20)
                if not top: continue
                w_words, w_counts = zip(*top)
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.barh(list(reversed(w_words)), list(reversed(w_counts)), color=color)
                ax.set_xlabel("Frequency"); ax.set_title(f"Top words — {label_name.title()}", fontweight="bold")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.patch.set_facecolor("#f8f7ff"); ax.set_facecolor("#f8f7ff")
                plt.tight_layout(); b64 = fig_to_b64(fig); plt.close(fig)
                if label_name == "positive": wc_pos_b64 = b64
                else: wc_neg_b64 = b64

        return jsonify({
            "message": f"✅ {MODEL_NAMES[clf_name]} trained successfully",
            "accuracy": acc,
            "report": report,
            "cm_chart": cm_b64,
            "wc_pos": wc_pos_b64,
            "wc_neg": wc_neg_b64,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": f"❌ Error: {e}"}), 500


@app.route("/api/eda")
def eda():
    df = get_demo_df()
    counts = df["text_label"].value_counts()

    # Distribution chart
    dist_b64 = build_distribution_chart(df)

    # VADER agreement
    df["vader"] = df["text"].apply(vader_predict)
    agreement = (df["vader"] == df["text_label"]).mean()

    # VADER vs true labels bar chart
    from collections import Counter
    fig, ax = plt.subplots(figsize=(7, 3))
    colours = {"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#fbbf24"}
    labels_order = ["Positive", "Negative", "Neutral"]
    x = np.arange(len(labels_order)); width = 0.35
    vader_cnts = [int((df["vader"] == l).sum()) for l in labels_order]
    true_cnts  = [int((df["text_label"] == l).sum()) for l in labels_order]
    ax.bar(x - width/2, true_cnts,  width, label="True Label", color=[colours[l] for l in labels_order], alpha=0.9)
    ax.bar(x + width/2, vader_cnts, width, label="VADER Prediction", color=[colours[l] for l in labels_order], alpha=0.5, edgecolor="#9ca3af", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels_order)
    ax.set_ylabel("Count"); ax.set_title("VADER Predictions vs True Labels", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#f8f7ff"); ax.set_facecolor("#f8f7ff")
    plt.tight_layout()
    vader_b64 = fig_to_b64(fig); plt.close(fig)

    # Sample rows
    sample = df[["text", "text_label"]].sample(10, random_state=7)
    sample["words"] = sample["text"].str.split().str.len()
    samples = sample.rename(columns={"text_label": "label"})[["text", "label", "words"]].to_dict("records")

    return jsonify({
        "total": len(df),
        "positive": int(counts.get("Positive", 0)),
        "negative": int(counts.get("Negative", 0)),
        "neutral":  int(counts.get("Neutral", 0)),
        "dist_chart": dist_b64,
        "vader_agreement": agreement,
        "vader_chart": vader_b64,
        "samples": samples,
    })


if __name__ == "__main__":
    print("🧠 Sentiment Analysis System — CSC 309 Mini Project #2")
    print("   Starting server at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
