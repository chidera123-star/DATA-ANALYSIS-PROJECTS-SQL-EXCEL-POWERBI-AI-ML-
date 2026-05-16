"""
Machine Translation System — CSC 309 Project #14
Streamlit App — EN → FR / ES / DE using Helsinki-NLP MarianMT Transformers
Author: Okoro-Enyi Reginald | FUTO School of Computing
"""

import streamlit as st
import time
from translate import load_model, translate_text, MODELS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Translation — CSC 309 P14",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.5rem 0 0; opacity: 0.75; font-size: 0.95rem; }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        flex: 1;
        text-align: center;
    }
    .metric-card .label { font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #0f3460; }

    /* Translation box */
    .translation-box {
        background: #f0fdf4;
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        font-size: 1.05rem;
        color: #14532d;
        min-height: 80px;
        line-height: 1.6;
    }
    .translation-box.empty {
        color: #86efac;
        font-style: italic;
    }

    /* BLEU badge */
    .bleu-badge {
        display: inline-block;
        background: #0f3460;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .quality-pill {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Sidebar */
    .sidebar-section { margin-bottom: 1.25rem; }
    .sidebar-section h4 { color: #0f3460; margin-bottom: 0.4rem; }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        padding-top: 1rem;
    }
    .stTextArea textarea { font-size: 1rem !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
LANG_META = {
    "French 🇫🇷":   {"code": "fr", "flag": "🇫🇷", "bleu": "42–48", "model": "Helsinki-NLP/opus-mt-en-fr"},
    "Spanish 🇪🇸":  {"code": "es", "flag": "🇪🇸", "bleu": "43–49", "model": "Helsinki-NLP/opus-mt-en-es"},
    "German 🇩🇪":   {"code": "de", "flag": "🇩🇪", "bleu": "38–44", "model": "Helsinki-NLP/opus-mt-en-de"},
}

BLEU_LABELS = {
    "fr": ("High Quality", "#16a34a"),
    "es": ("High Quality", "#16a34a"),
    "de": ("Good Quality", "#2563eb"),
}

EXAMPLES = [
    "Machine learning is transforming the world.",
    "Good morning, how are you today?",
    "Artificial intelligence is the future of technology.",
    "I would like to order a coffee, please.",
    "The Federal University of Technology Owerri trains world-class engineers.",
    "Neural networks learn from large amounts of data.",
]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    target_lang = st.selectbox(
        "Target Language",
        list(LANG_META.keys()),
        index=0,
        help="Select the language to translate English text into.",
    )
    meta = LANG_META[target_lang]

    st.markdown("---")
    st.markdown("### 📋 Model Info")
    st.markdown(f"**Model:** `{meta['model']}`")
    st.markdown(f"**Expected BLEU:** `{meta['bleu']}`")
    st.markdown(f"**Beam Search:** k = 5")
    st.markdown(f"**Max tokens:** 128")

    st.markdown("---")
    st.markdown("### 📚 Quick Examples")
    for i, ex in enumerate(EXAMPLES):
        if st.button(f"📝 Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state["input_text"] = ex

    st.markdown("---")
    st.markdown("### 📊 BLEU Score Guide")
    bleu_rows = [
        ("< 10",  "Unusable"),
        ("10–19", "Hard to understand"),
        ("20–29", "Understandable"),
        ("30–40", "Good quality"),
        ("> 40",  "High quality ✓"),
    ]
    for score, desc in bleu_rows:
        st.markdown(f"**{score}** — {desc}")

# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🌐 Machine Translation System</h1>
  <p>CSC 309 — AI Mini Project #14 &nbsp;|&nbsp; EN → FR / ES / DE &nbsp;|&nbsp;
     Helsinki-NLP MarianMT Transformers &nbsp;|&nbsp; Okoro-Enyi Reginald · FUTO</p>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Architecture</div>
    <div class="value" style="font-size:1.1rem;">MarianMT Transformer</div>
  </div>
  <div class="metric-card">
    <div class="label">Language Pairs</div>
    <div class="value">EN → 3</div>
  </div>
  <div class="metric-card">
    <div class="label">Training Corpus</div>
    <div class="value" style="font-size:1.1rem;">OPUS (1M+ pairs)</div>
  </div>
  <div class="metric-card">
    <div class="label">Target BLEU</div>
    <div class="value">> 40</div>
  </div>
  <div class="metric-card">
    <div class="label">Beam Search k</div>
    <div class="value">5</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Translation UI ─────────────────────────────────────────────────────────────
col_src, col_tgt = st.columns(2, gap="large")

with col_src:
    st.markdown(f"#### 🇬🇧 English (Source)")
    input_key = "input_text"
    input_text = st.text_area(
        label="English input",
        value=st.session_state.get(input_key, ""),
        height=180,
        placeholder="Type or paste an English sentence here…",
        label_visibility="collapsed",
        key=input_key,
    )
    char_count = len(input_text)
    st.caption(f"{char_count} characters")

    translate_btn = st.button(
        f"Translate to {target_lang}  →",
        type="primary",
        use_container_width=True,
        disabled=(char_count == 0),
    )

with col_tgt:
    st.markdown(f"#### {meta['flag']} {target_lang.split()[0]} (Target)")

    result_placeholder = st.empty()
    info_placeholder   = st.empty()

    if "translation" not in st.session_state:
        st.session_state["translation"] = ""
        st.session_state["elapsed"]     = 0.0
        st.session_state["last_lang"]   = ""
        st.session_state["last_input"]  = ""

    # Show current translation or placeholder
    trans = st.session_state["translation"]
    if trans:
        result_placeholder.markdown(
            f'<div class="translation-box">{trans}</div>',
            unsafe_allow_html=True,
        )
        elapsed = st.session_state["elapsed"]
        ql, qc = BLEU_LABELS[meta["code"]]
        info_placeholder.markdown(
            f'<span class="bleu-badge">BLEU {meta["bleu"]}</span> &nbsp;'
            f'<span class="quality-pill" style="background:{qc}22;color:{qc};">{ql}</span>'
            f'&nbsp;&nbsp; ⏱ {elapsed:.2f}s',
            unsafe_allow_html=True,
        )
    else:
        result_placeholder.markdown(
            '<div class="translation-box empty">Translation will appear here…</div>',
            unsafe_allow_html=True,
        )

# ── Translation logic ──────────────────────────────────────────────────────────
if translate_btn and input_text.strip():
    lang_code = meta["code"]

    with col_tgt:
        result_placeholder.markdown(
            '<div class="translation-box empty">⏳ Loading model & translating…</div>',
            unsafe_allow_html=True,
        )

    with st.spinner(f"Loading Helsinki-NLP/opus-mt-en-{lang_code} and translating…"):
        try:
            t0  = time.time()
            tok, model = load_model(lang_code)
            out = translate_text(input_text.strip(), lang_code, tok, model)
            elapsed = time.time() - t0

            st.session_state["translation"] = out
            st.session_state["elapsed"]     = elapsed
            st.session_state["last_lang"]   = lang_code
            st.session_state["last_input"]  = input_text

            with col_tgt:
                result_placeholder.markdown(
                    f'<div class="translation-box">{out}</div>',
                    unsafe_allow_html=True,
                )
                ql, qc = BLEU_LABELS[lang_code]
                info_placeholder.markdown(
                    f'<span class="bleu-badge">BLEU {meta["bleu"]}</span> &nbsp;'
                    f'<span class="quality-pill" style="background:{qc}22;color:{qc};">{ql}</span>'
                    f'&nbsp;&nbsp; ⏱ {elapsed:.2f}s',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            with col_tgt:
                result_placeholder.error(f"Translation failed: {e}")

# ── Architecture diagram ───────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📐 System Architecture & Pipeline", expanded=False):
    st.markdown("""
```
INPUT (English Text)
        │
        ▼
┌───────────────────────┐
│  MarianTokenizer      │  ← sentencepiece BPE tokenisation
│  (sentencepiece BPE)  │
└──────────┬────────────┘
           │  token ids + attention mask
           ▼
┌───────────────────────────────────────────────────────┐
│           Helsinki-NLP MarianMT Transformer            │
│                                                       │
│  Encoder (6 layers, 512 dim, 8 heads)                 │
│   ↓ self-attention over all source tokens             │
│  Decoder (6 layers, 512 dim, 8 heads)                 │
│   ↓ cross-attention to encoder hidden states          │
│  Beam Search  k=5  max_len=128                        │
└──────────┬────────────────────────────────────────────┘
           │  generated token ids
           ▼
┌───────────────────────┐
│  Decode + detokenise  │  ← skip_special_tokens=True
└──────────┬────────────┘
           │
           ▼
     Translated Text
```

| Stage | Detail |
|---|---|
| Tokeniser | MarianTokenizer (sentencepiece BPE) |
| Model | Helsinki-NLP/opus-mt-en-{fr\|es\|de} |
| Encoder | 6-layer transformer, 512 dim, 8 heads |
| Decoder | 6-layer transformer with cross-attention |
| Decoding | Beam Search k=5, max_length=128 |
| Training corpus | OPUS (Helsinki-NLP), 1M+ parallel pairs |
| Expected BLEU | 42–48 (EN→FR), 43–49 (EN→ES), 38–44 (EN→DE) |
""")

# ── Model comparison table ─────────────────────────────────────────────────────
with st.expander("📊 BLEU Score Benchmark", expanded=False):
    st.markdown("""
| Model | BLEU EN→FR | Notes |
|---|---|---|
| Word-by-Word Dictionary | ~5 | No grammar; word substitution only |
| Statistical MT (phrase-based) | ~25 | Phrase tables; limited context |
| Custom LSTM Seq2Seq (Tatoeba 10k) | ~28–32 | Attention mechanism |
| **Helsinki-NLP MarianMT (this app)** | **~42–48** | **OPUS corpus; used here** |
| Google Translate (reference) | ~55 | Massive data + proprietary model |

> **This app uses the pretrained MarianMT path** — the highest-quality option documented in the PDR,
> achieving BLEU > 40 with zero additional training on your hardware.
""")

# ── About ──────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this Project", expanded=False):
    st.markdown("""
**Machine Translation System** is CSC 309 Mini Project #14 at the  
Federal University of Technology Owerri — School of Computing.

The system demonstrates:
- **Neural Machine Translation** using Seq2Seq encoder-decoder architectures  
- **Transformer architecture** via Hugging Face Helsinki-NLP/opus-mt pretrained models  
- **Attention mechanism** — multi-head self-attention + cross-attention  
- **Beam search decoding** (k=5) for higher-quality output vs greedy decoding  
- **BLEU score evaluation** as the standard MT quality metric  

**Languages supported:** English → French · Spanish · German  
**Author:** Okoro-Enyi Reginald | **Course:** CSC 309 — Artificial Intelligence  
**Document:** PDR-MT-CSC309-014 v1.0 | 10 May 2026
""")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  CSC 309 — Artificial Intelligence Mini Projects &nbsp;|&nbsp;
  Project #14: Machine Translation System &nbsp;|&nbsp;
  Okoro-Enyi Reginald &nbsp;|&nbsp;
  Federal University of Technology Owerri &nbsp;|&nbsp; 2026
</div>
""", unsafe_allow_html=True)
