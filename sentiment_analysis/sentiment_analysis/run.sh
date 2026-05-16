#!/bin/bash
# ─────────────────────────────────────────────────────────────
# CSC 309 Mini Project #2 — Sentiment Analysis System
# Run script: installs deps (if needed) then starts the server
# ─────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

echo "📦 Checking dependencies..."
pip install flask scikit-learn pandas matplotlib joblib numpy -q

echo ""
echo "🧠 Sentiment Analysis System — CSC 309 Mini Project #2"
echo "   Open http://localhost:5000 in your browser"
echo ""

# Flask mode (default) — works without NLTK/Streamlit
python3 flask_app.py
