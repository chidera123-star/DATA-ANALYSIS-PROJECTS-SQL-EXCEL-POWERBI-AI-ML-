"""train.py — Build and train Scikit-learn pipelines for sentiment analysis."""
import os
import time
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def build_pipeline(clf_name: str = "lr") -> Pipeline:
    """Return a TF-IDF + classifier pipeline."""
    clf = {
        "nb": MultinomialNB(),
        "lr": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
        "svm": LinearSVC(C=1.0, max_iter=2000),
    }[clf_name]

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    strip_accents="unicode",
                ),
            ),
            ("clf", clf),
        ]
    )


def train(train_df, clf_name: str = "lr") -> Pipeline:
    """Fit the pipeline and save to outputs/model_{clf_name}.pkl."""
    pipe = build_pipeline(clf_name)
    t0 = time.time()
    pipe.fit(train_df["text"], train_df["label"])
    elapsed = time.time() - t0
    print(f"[{clf_name.upper()}] Trained in {elapsed:.1f}s")
    out_path = os.path.join(OUTPUTS_DIR, f"model_{clf_name}.pkl")
    joblib.dump(pipe, out_path)
    print(f"Saved → {out_path}")
    return pipe


def load_model(clf_name: str = "lr") -> Pipeline:
    """Load a saved pipeline from disk."""
    path = os.path.join(OUTPUTS_DIR, f"model_{clf_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train() first."
        )
    return joblib.load(path)
