"""data/load_data.py — Hugging Face IMDB dataset loader + local CSV fallback."""
import os
import pandas as pd
from preprocessing.clean import clean_text

DEMO_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "demo_samples.csv")


def load_imdb(max_samples: int = 5000):
    """
    Load the IMDB 50k dataset via Hugging Face.
    Returns (train_df, test_df) with columns ['text', 'label'].
    Falls back to demo CSV if Hugging Face is unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("imdb")
        train_df = pd.DataFrame(ds["train"]).sample(
            min(max_samples, len(ds["train"])), random_state=42
        )
        test_df = pd.DataFrame(ds["test"]).sample(
            min(max_samples // 5, len(ds["test"])), random_state=42
        )
        train_df["text"] = train_df["text"].apply(clean_text)
        test_df["text"] = test_df["text"].apply(clean_text)
        return train_df[["text", "label"]], test_df[["text", "label"]]
    except Exception:
        return load_demo()


def load_demo():
    """Load built-in demo CSV (Pos/Neg/Neutral, ~500 samples)."""
    df = pd.read_csv(DEMO_CSV)
    df["text"] = df["text"].apply(clean_text)
    # Map string labels to ints: Positive=1, Negative=0, Neutral=2
    label_map = {"Positive": 1, "Negative": 0, "Neutral": 2}
    df["label"] = df["text_label"].map(label_map)
    split = int(len(df) * 0.8)
    return df.iloc[:split][["text", "label"]], df.iloc[split:][["text", "label"]]
