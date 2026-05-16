"""evaluate.py — Accuracy, classification report, confusion matrix, ROC, word clouds."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_labels(n_classes: int):
    if n_classes == 2:
        return ["Negative", "Positive"]
    return ["Negative", "Positive", "Neutral"]


def evaluate(pipe, test_df) -> dict:
    """
    Evaluate pipeline on test_df.
    Returns dict with accuracy, report string, and paths to saved figures.
    """
    y_pred = pipe.predict(test_df["text"])
    y_true = test_df["label"]
    n_classes = len(np.unique(y_true))
    labels = get_labels(n_classes)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels)

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        cmap="Purples", ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── ROC curve (binary only) ────────────────────────────────────────
    roc_path = None
    if n_classes == 2 and hasattr(pipe, "predict_proba"):
        try:
            y_score = pipe.predict_proba(test_df["text"])[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.plot(fpr, tpr, color="#7C3AED", lw=2,
                     label=f"ROC curve (AUC = {roc_auc:.3f})")
            ax2.plot([0, 1], [0, 1], "k--", lw=1)
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve", fontsize=14, fontweight="bold")
            ax2.legend(loc="lower right")
            roc_path = os.path.join(OUTPUTS_DIR, "roc_curve.png")
            fig2.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
        except Exception:
            pass

    # ── Save text report ──────────────────────────────────────────────
    report_path = os.path.join(OUTPUTS_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    return {
        "accuracy": acc,
        "report": report,
        "cm_path": cm_path,
        "roc_path": roc_path,
        "report_path": report_path,
    }


def generate_wordclouds(train_df):
    """Generate positive and negative word-cloud images."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None, None

    paths = {}
    for label_int, label_name in {1: "positive", 0: "negative"}.items():
        subset = train_df[train_df["label"] == label_int]["text"]
        text_blob = " ".join(subset.tolist())
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="RdYlGn" if label_int == 1 else "Reds",
            max_words=150,
        ).generate(text_blob)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"Most Frequent Words — {label_name.capitalize()} Reviews",
            fontsize=14, fontweight="bold"
        )
        path = os.path.join(OUTPUTS_DIR, f"wordcloud_{label_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[label_name] = path

    return paths.get("positive"), paths.get("negative")
