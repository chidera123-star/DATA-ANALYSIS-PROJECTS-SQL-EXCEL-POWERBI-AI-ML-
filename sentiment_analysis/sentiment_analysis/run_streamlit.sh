#!/bin/bash
# Run the Streamlit version (requires: pip install streamlit nltk)
cd "$(dirname "$0")"
pip install streamlit nltk scikit-learn pandas matplotlib wordcloud joblib plotly -q
streamlit run app.py
