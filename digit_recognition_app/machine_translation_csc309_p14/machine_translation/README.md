# 🌐 Machine Translation System
**CSC 309 — AI Mini Project #14**  
Okoro-Enyi Reginald | Federal University of Technology Owerri | 2026

---

## Overview

A Streamlit web app that translates English text into **French**, **Spanish**, and **German**  
using the **Helsinki-NLP/opus-mt** pretrained MarianMT transformer models from Hugging Face.

| Item | Detail |
|---|---|
| Architecture | MarianMT Transformer (6 layers, 512 dim, 8 heads) |
| Languages | EN → FR · EN → ES · EN → DE |
| Decoding | Beam Search k=5, max_length=128 |
| Expected BLEU | 42–48 (FR), 43–49 (ES), 38–44 (DE) |
| Framework | Streamlit + Hugging Face Transformers + PyTorch |

---

## Project Structure

```
machine_translation/
├── app.py              ← Streamlit entry point
├── translate.py        ← Model loading & translation logic
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## Setup & Run

### 1. Clone / unzip the project
```bash
cd machine_translation
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** PyTorch (~800 MB) and the three Helsinki-NLP models (~300 MB each)  
> will be downloaded on first run and cached in `~/.cache/huggingface/`.  
> Subsequent runs are fast. An internet connection is required the first time.

### 4. Launch the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## Usage

1. Type or paste an English sentence in the left panel.  
2. Select a target language (French / Spanish / German) from the sidebar.  
3. Click **Translate →**.  
4. The translation appears in the right panel with timing and quality info.  
5. Use **Quick Examples** in the sidebar to try sample sentences instantly.

---

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch, and `app.py` as the main file.
4. Click **Deploy** — done. Your app gets a public HTTPS URL.

> Streamlit Cloud handles `requirements.txt` automatically.  
> Models are downloaded on first cold start (~5 min), then cached per session.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥ 1.35 | Web UI framework |
| transformers | ≥ 4.40 | Hugging Face MarianMT |
| torch | ≥ 2.2 | PyTorch backend |
| sentencepiece | ≥ 0.1.99 | BPE tokeniser for MarianMT |
| sacrebleu | ≥ 2.3 | BLEU evaluation (optional CLI use) |

---

## References

- Vaswani et al. (2017) — *Attention Is All You Need*
- Helsinki-NLP OPUS-MT — https://huggingface.co/Helsinki-NLP
- Hugging Face Transformers — https://huggingface.co/docs/transformers
- OPUS Parallel Corpus — https://opus.nlpl.eu
- PDR Document: PDR-MT-CSC309-014 v1.0 | 10 May 2026

---

*Federal University of Technology Owerri — School of Computing — CSC 309*
