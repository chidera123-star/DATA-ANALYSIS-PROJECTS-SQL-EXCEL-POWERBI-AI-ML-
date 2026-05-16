# 🧠 Sentiment Analysis System
**CSC 309 — Artificial Intelligence Mini Project #2**  
*Okoro-Enyi Reginald · Federal University of Technology Owerri · May 2026*

---

## Quick Start

### Option A — Flask (works everywhere, no extra installs needed)
```bash
pip install flask scikit-learn pandas matplotlib joblib numpy
python flask_app.py
# Open http://localhost:5000
```

### Option B — Streamlit (if you have Streamlit installed)
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure
```
sentiment_analysis/
├── flask_app.py          ← Flask web interface (primary)
├── app.py                ← Streamlit web interface (alternative)
├── train.py              ← Train TF-IDF + classifier pipelines
├── evaluate.py           ← Accuracy, confusion matrix, ROC, word clouds
├── preprocessing/
│   ├── clean.py          ← Text preprocessing (pure Python, no NLTK needed)
│   └── load_data.py      ← Dataset loader (HuggingFace + local CSV fallback)
├── baseline/
│   └── vader_baseline.py ← VADER-like lexicon sentiment (no NLTK needed)
├── data/
│   └── demo_samples.csv  ← 1,300 labelled Pos/Neg/Neutral samples
├── outputs/
│   ├── model_lr.pkl      ← Trained Logistic Regression + TF-IDF
│   ├── model_nb.pkl      ← Trained Naive Bayes + TF-IDF
│   └── model_svm.pkl     ← Trained Linear SVM + TF-IDF
├── requirements.txt      ← All dependencies
├── run.sh                ← One-command Flask startup
└── run_streamlit.sh      ← One-command Streamlit startup
```

---

## Features

| Page | Description |
|------|-------------|
| 🔍 **Live Prediction** | Real-time sentiment for any text, showing ML confidence + VADER comparison |
| 🏋️ **Train & Evaluate** | Train any of 3 classifiers, view confusion matrix, classification report, word clouds |
| 📊 **EDA & Charts** | Class distribution, text length histogram, VADER agreement analysis |
| 📖 **About** | Pipeline architecture, design decisions, future work |

## Models

| Model | Expected Accuracy |
|-------|------------------|
| VADER (rule-based) | 68–72% |
| Multinomial Naive Bayes | 83–85% |
| **Logistic Regression** ★ | **87–89%** |
| Linear SVM | 88–90% |

## Deploy on Streamlit Cloud
1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file to `app.py`
4. Done!
