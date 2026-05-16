"""
Handwritten Digit Recognition — Streamlit Web App
CSC 309 Mini Project #5 |  FUTO 2026
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageOps
import io
import os
import time

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recogniser · CSC 309",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.sub-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0;
}

.metric-card {
    background: #13131f;
    border: 1px solid #1e1e32;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}

.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.predict-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #a78bfa44;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

.digit-result {
    font-family: 'Space Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    color: #a78bfa;
    line-height: 1;
}

.confidence-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: #34d399;
}

.info-box {
    background: #13131f;
    border-left: 3px solid #a78bfa;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #9ca3af;
    margin: 0.5rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px #7c3aed55;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
}

div[data-testid="stSidebar"] {
    background: #0d0d1a;
    border-right: 1px solid #1e1e32;
}

.arch-layer {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #9ca3af;
    margin: 2px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Load / Train Model (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """Train CNN on MNIST and cache it in session."""
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    # Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test,  10)

    # Val split
    split = int(len(X_train) * 0.8)
    X_val, y_val   = X_train[split:], y_train_cat[split:]
    X_train_s, y_train_s = X_train[:split], y_train_cat[:split]

    # Build CNN
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1), padding="valid"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu", padding="valid"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(
        X_train_s, y_train_s,
        epochs=15, batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[es], verbose=0
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    return model, history, test_acc, test_loss, X_test, y_test


def preprocess_canvas_image(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image (any size) to a 28×28 float32 array ready for the model."""
    img = pil_img.convert("L")           # greyscale
    img = ImageOps.invert(img)           # white digit on black → black bg, white digit
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    # Auto-contrast: rescale so max pixel = 1
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr.reshape(1, 28, 28, 1)


def plot_probability_bar(probs: np.ndarray):
    """Return a matplotlib figure of per-class probabilities."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#13131f")
    ax.set_facecolor("#13131f")

    colors = ["#a78bfa" if i == np.argmax(probs) else "#2d2d4a" for i in range(10)]
    bars = ax.bar(range(10), probs * 100, color=colors, width=0.6, zorder=3)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)],
                       fontsize=11, color="#e8e8f0", fontfamily="monospace")
    ax.set_ylabel("Confidence (%)", fontsize=9, color="#6b7280")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="y", colors="#6b7280", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#1e1e32", linewidth=0.8, zorder=0)

    for bar, p in zip(bars, probs):
        if p > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{p*100:.1f}%", ha="center", va="bottom",
                    fontsize=7, color="#9ca3af", fontfamily="monospace")

    ax.set_title("Class Probability Distribution",
                 fontsize=10, color="#9ca3af", pad=8, fontfamily="monospace")
    plt.tight_layout()
    return fig


def plot_training_curves(history):
    """Return a matplotlib figure of train/val accuracy & loss."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    fig.patch.set_facecolor("#0a0a0f")
    for ax in axes:
        ax.set_facecolor("#13131f")
        ax.spines[:].set_color("#1e1e32")
        ax.tick_params(colors="#6b7280")

    h = history.history
    ep = range(1, len(h["accuracy"]) + 1)

    axes[0].plot(ep, h["accuracy"],     color="#a78bfa", lw=2, label="Train")
    axes[0].plot(ep, h["val_accuracy"], color="#60a5fa", lw=2, ls="--", label="Val")
    axes[0].set_title("Accuracy", color="#9ca3af", fontsize=10)
    axes[0].set_xlabel("Epoch", color="#6b7280", fontsize=8)
    axes[0].legend(fontsize=8, framealpha=0)
    axes[0].yaxis.grid(True, color="#1e1e32", lw=0.8)

    axes[1].plot(ep, h["loss"],     color="#f472b6", lw=2, label="Train")
    axes[1].plot(ep, h["val_loss"], color="#34d399", lw=2, ls="--", label="Val")
    axes[1].set_title("Loss", color="#9ca3af", fontsize=10)
    axes[1].set_xlabel("Epoch", color="#6b7280", fontsize=8)
    axes[1].legend(fontsize=8, framealpha=0)
    axes[1].yaxis.grid(True, color="#1e1e32", lw=0.8)

    plt.tight_layout()
    return fig


def show_sample_grid(X_test, y_test, model):
    """Return a figure showing 20 test samples with predicted labels."""
    indices = np.random.choice(len(X_test), 20, replace=False)
    preds = np.argmax(model.predict(X_test[indices], verbose=0), axis=1)
    trues = y_test[indices]

    fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
    fig.patch.set_facecolor("#0a0a0f")
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[indices[i]].reshape(28,28), cmap="inferno")
        correct = preds[i] == trues[i]
        ax.set_title(f"P:{preds[i]}", fontsize=8,
                     color="#34d399" if correct else "#f87171",
                     fontfamily="monospace")
        ax.axis("off")
    plt.suptitle("Random Test Samples  |  green = correct  |  red = wrong",
                 fontsize=9, color="#6b7280", y=1.02)
    plt.tight_layout()
    return fig


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title">🔢 MNIST</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">CSC 309 · Project #5</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Student**")
    st.markdown("Okoro-Enyi Reginald")
    st.markdown("Federal University of Technology Owerri")

    st.markdown("---")
    st.markdown("**CNN Architecture**")
    layers = [
        "Input  28×28×1",
        "Conv2D  32 filters 3×3",
        "BatchNorm + MaxPool",
        "Conv2D  64 filters 3×3",
        "BatchNorm + MaxPool",
        "Flatten → 1600",
        "Dense  128  ReLU",
        "Dropout  0.5",
        "Dense  10  Softmax",
    ]
    for l in layers:
        st.markdown(f'<div class="arch-layer">▸ {l}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("MNIST · 70,000 images · 10 classes")
    st.markdown("Train: 48k · Val: 12k · Test: 10k")

    st.markdown("---")
    st.markdown('<span style="font-size:0.7rem;color:#4b5563;font-family:monospace">Built with TensorFlow + Streamlit</span>', unsafe_allow_html=True)


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">Handwritten Digit Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Deep Learning with MNIST · TensorFlow CNN · CSC 309 Mini Project #5</p>', unsafe_allow_html=True)
st.markdown("")

# ─── Train / Load model ───────────────────────────────────────────────────────
with st.spinner("🧠 Training CNN on MNIST (first run only — ~2 min on CPU)…"):
    model, history, test_acc, test_loss, X_test, y_test = load_or_train_model()

# ─── Metrics row ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{test_acc*100:.2f}%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{test_loss:.4f}</div><div class="metric-label">Test Loss</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">70K</div><div class="metric-label">Dataset Size</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">~225K</div><div class="metric-label">Parameters</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🖊️  Predict", "📈  Training Results", "🔬  Test Samples"])

# ═══════════════════════════ TAB 1: PREDICT ═══════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Upload a digit image")
        st.markdown('<div class="info-box">Upload a clear image of a single handwritten digit (0–9). Works best with white digit on dark background, or dark digit on white background.</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("#### Or draw a digit")
        st.markdown('<div class="info-box">Use the canvas below to sketch a digit, then click Predict.</div>', unsafe_allow_html=True)

        # HTML5 canvas drawing widget
        canvas_html = """
        <style>
        #digit-canvas {
            border: 2px solid #a78bfa55;
            border-radius: 12px;
            cursor: crosshair;
            background: #000;
            display: block;
        }
        .canvas-btns { display:flex; gap:8px; margin-top:8px; }
        .canvas-btns button {
            flex:1; padding:8px; border:none; border-radius:6px;
            font-family:'Space Mono',monospace; font-size:12px;
            cursor:pointer; font-weight:700;
        }
        #predict-btn { background: linear-gradient(135deg,#7c3aed,#4f46e5); color:#fff; }
        #clear-btn   { background: #1e1e32; color:#9ca3af; }
        #output-img  { display:none; }
        </style>

        <canvas id="digit-canvas" width="280" height="280"></canvas>
        <div class="canvas-btns">
            <button id="predict-btn" onclick="sendCanvas()">▶ Predict</button>
            <button id="clear-btn"   onclick="clearCanvas()">✕ Clear</button>
        </div>
        <img id="output-img"/>

        <script>
        const canvas = document.getElementById('digit-canvas');
        const ctx    = canvas.getContext('2d');
        ctx.fillStyle = '#000';
        ctx.fillRect(0,0,280,280);
        let drawing = false;

        function getPos(e) {
            const r = canvas.getBoundingClientRect();
            if (e.touches) {
                return { x: e.touches[0].clientX - r.left,
                         y: e.touches[0].clientY - r.top };
            }
            return { x: e.clientX - r.left, y: e.clientY - r.top };
        }

        canvas.addEventListener('mousedown',  e => { drawing=true; draw(e); });
        canvas.addEventListener('mousemove',  e => { if(drawing) draw(e); });
        canvas.addEventListener('mouseup',    () => drawing=false);
        canvas.addEventListener('mouseleave', () => drawing=false);
        canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing=true; draw(e); });
        canvas.addEventListener('touchmove',  e => { e.preventDefault(); if(drawing) draw(e); });
        canvas.addEventListener('touchend',   () => drawing=false);

        function draw(e) {
            const p = getPos(e);
            ctx.beginPath();
            ctx.arc(p.x, p.y, 12, 0, Math.PI*2);
            ctx.fillStyle = '#fff';
            ctx.fill();
        }

        function clearCanvas() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0,0,280,280);
        }

        function sendCanvas() {
            const dataUrl = canvas.toDataURL('image/png');
            // Store in session via a hidden element for Streamlit to read
            document.getElementById('output-img').src = dataUrl;
            // Copy to clipboard as a signal (not reliable) — use file_uploader workaround
            const link = document.createElement('a');
            link.download = 'drawn_digit.png';
            link.href = dataUrl;
            link.click();
        }
        </script>
        """
        st.components.v1.html(canvas_html, height=360)
        st.markdown('<div class="info-box">💡 After drawing, click <b>▶ Predict</b> to download the image, then upload it above.</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Prediction Result")

        if uploaded is not None:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", width=140)

            arr = preprocess_canvas_image(img)
            probs = model.predict(arr, verbose=0)[0]
            pred_digit = int(np.argmax(probs))
            confidence = float(probs[pred_digit]) * 100

            conf_color = "#34d399" if confidence > 80 else "#fbbf24" if confidence > 50 else "#f87171"

            st.markdown(f"""
            <div class="predict-box">
                <div style="color:#6b7280;font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;font-family:monospace">Predicted Digit</div>
                <div class="digit-result">{pred_digit}</div>
                <div class="confidence-text" style="color:{conf_color}">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(confidence / 100)

            st.markdown("**Class probabilities**")
            fig = plot_probability_bar(probs)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Top-3
            top3 = np.argsort(probs)[::-1][:3]
            st.markdown("**Top-3 candidates**")
            for rank, cls in enumerate(top3):
                st.markdown(
                    f'<div class="info-box">#{rank+1}  Digit <b style="color:#a78bfa">{cls}</b> — {probs[cls]*100:.2f}%</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown("""
            <div class="predict-box" style="opacity:0.5;">
                <div style="color:#6b7280;font-size:0.75rem;letter-spacing:2px;font-family:monospace">AWAITING INPUT</div>
                <div class="digit-result" style="color:#2d2d4a">?</div>
                <div style="color:#4b5563;font-size:0.8rem">Upload or draw a digit</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="info-box">📂 Upload any image of a handwritten digit (0–9) to see the model predict it in real time.</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">🖊️ Or use the canvas on the left to draw, then download and upload the result.</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">✅ Model achieves <b style="color:#a78bfa">98.5%+</b> accuracy on the official MNIST test set.</div>', unsafe_allow_html=True)


# ═══════════════════════════ TAB 2: TRAINING ══════════════════════════════════
with tab2:
    st.markdown("#### Training & Validation Curves")
    fig_curves = plot_training_curves(history)
    st.pyplot(fig_curves, use_container_width=True)
    plt.close(fig_curves)

    final_acc = history.history["accuracy"][-1]
    final_val = history.history["val_accuracy"][-1]
    epochs_ran = len(history.history["accuracy"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{final_acc*100:.1f}%</div><div class="metric-label">Final Train Acc</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{final_val*100:.1f}%</div><div class="metric-label">Final Val Acc</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{epochs_ran}</div><div class="metric-label">Epochs Trained</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Training Configuration")
    cfg = {
        "Optimiser": "Adam (lr=0.001)",
        "Loss Function": "Categorical Cross-Entropy",
        "Batch Size": "128",
        "Max Epochs": "15 (Early Stopping patience=3)",
        "Train / Val Split": "80% / 20% of 60k samples",
        "Regularisation": "BatchNorm + Dropout(0.5)",
    }
    for k, v in cfg.items():
        st.markdown(f'<div class="info-box"><b style="color:#a78bfa">{k}</b>: {v}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Benchmark Comparison")
    benchmark = {
        "Logistic Regression": "~92.5%",
        "Dense Network (MLP)": "~97.5%",
        "**This CNN (Project)**": f"**{test_acc*100:.1f}%**",
        "CNN + Data Augmentation": "~99.2%",
        "State-of-the-Art": "~99.8%",
    }
    for m, a in benchmark.items():
        color = "#a78bfa" if "Project" in m else "#6b7280"
        st.markdown(f'<div class="info-box" style="border-color:{color}"><b style="color:{color}">{m}</b>: {a}</div>', unsafe_allow_html=True)


# ═══════════════════════════ TAB 3: TEST SAMPLES ══════════════════════════════
with tab3:
    st.markdown("#### Random Test Set Predictions")
    st.markdown('<div class="info-box">20 randomly selected samples from the MNIST test set. Green title = correct prediction, Red = misclassified.</div>', unsafe_allow_html=True)

    if st.button("🔀  Refresh Random Sample"):
        st.cache_resource.clear()

    fig_samples = show_sample_grid(X_test, y_test, model)
    st.pyplot(fig_samples, use_container_width=True)
    plt.close(fig_samples)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Common Confusion Pairs")
    confusions = [
        ("4 ↔ 9", "Similar loop structure in upper portion"),
        ("3 ↔ 5", "Overlapping curve shapes top and bottom"),
        ("7 ↔ 1", "Both predominantly vertical strokes"),
        ("8 ↔ 3", "Right-side curves visually similar"),
    ]
    for pair, reason in confusions:
        st.markdown(f'<div class="info-box"><b style="color:#f472b6">{pair}</b>: {reason}</div>', unsafe_allow_html=True)
