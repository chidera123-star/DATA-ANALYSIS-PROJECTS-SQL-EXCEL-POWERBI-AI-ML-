"""baseline/vader_baseline.py — Lightweight VADER-like lexicon (no NLTK required).

Uses a compact positive/negative word list with negation and intensifier handling.
For production, replace with: from nltk.sentiment.vader import SentimentIntensityAnalyzer
"""
import re
from typing import Dict

# Compact sentiment lexicons
_POS_WORDS = set("""
absolutely amazing awesome beautiful best better brilliant celebrated cheerful
classic clean clear clever complete confident creative cute dazzling dedicated
delightful distinguished dynamic effective efficient elegant enchanting
enjoyable enthusiastic excellent exceptional exciting exquisite extraordinary
fabulous fantastic fine flawless fresh friendly fun generous genius graceful
happy helpful hilarious honest impressive incredible innovative inspiring
interesting inventive joyful kind legendary lively love lovely magnificent
marvelous masterful memorable modern natural nice notable outstanding passionate
perfect phenomenal pleasant polished positive powerful precise professional
quick radiant reliable remarkable resplendent rich robust satisfying skilled
smooth sophisticated spectacular splendid strong stunning stylish sublime
superb superior talented terrific thorough timely unforgettable unique
upbeat valuable vibrant warm wonderful worthwhile
""".split())

_NEG_WORDS = set("""
abysmal annoying appalling arrogant atrocious awful bad boring broken cheap
confusing corrupt crappy crude damaged dangerous defective deficient demeaning
deplorable desperate difficult disappointing disgusting dreadful dull
dysfunctional embarrassing empty evil excessive exhausting expensive failed
fake flawed frustrating ghastly grim gross horrible hostile inadequate inferior
insignificant irritating lacking lame lousy mediocre misleading monotonous
nasty offensive overpriced pathetic pointless poor predictable problematic
ridiculous rude rubbish sad shallow shoddy slow subpar terrible toxic
ugly unacceptable unreliable unsafe useless vague vile wasted weak worthless
wrong
""".split())

_INTENSIFIERS = {"very", "extremely", "absolutely", "completely", "totally",
                 "utterly", "incredibly", "highly", "so", "really", "quite"}
_NEGATORS = {"not", "no", "never", "don't", "doesn't", "didn't", "won't",
             "can't", "cannot", "hardly", "barely", "scarcely"}


def vader_scores(text: str) -> Dict[str, float]:
    tokens = re.sub(r"[^a-zA-Z\s']", " ", text.lower()).split()
    pos, neg = 0.0, 0.0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Check for negation window (2 words back)
        negated = any(tokens[max(0, i-j)] in _NEGATORS for j in range(1, 3))
        # Check for intensifier (1 word back)
        intensified = i > 0 and tokens[i - 1] in _INTENSIFIERS
        weight = 1.5 if intensified else 1.0
        if tok in _POS_WORDS:
            if negated:
                neg += weight
            else:
                pos += weight
        elif tok in _NEG_WORDS:
            if negated:
                pos += weight * 0.5
            else:
                neg += weight
        i += 1

    total = pos + neg or 1.0
    pos_n = pos / total
    neg_n = neg / total
    neu_n = max(0.0, 1.0 - pos_n - neg_n)
    compound = (pos - neg) / (pos + neg + 0.001)
    compound = max(-1.0, min(1.0, compound))
    return {"pos": round(pos_n, 3), "neg": round(neg_n, 3),
            "neu": round(neu_n, 3), "compound": round(compound, 4)}


def vader_predict(text: str) -> str:
    s = vader_scores(text)
    if s["compound"] >= 0.05:
        return "Positive"
    elif s["compound"] <= -0.05:
        return "Negative"
    return "Neutral"


def vader_accuracy(test_df) -> float:
    label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    preds = test_df["text"].apply(vader_predict)
    true = test_df["label"].map(label_map)
    return (preds == true).mean()
