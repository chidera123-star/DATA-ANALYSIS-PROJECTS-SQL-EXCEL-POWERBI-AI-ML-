"""preprocessing/clean.py — Tokenise, lemmatise, stop-word removal (no NLTK required)."""
import re

STOPS = set(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are "
    "was were be been being have has had having do does did doing a an the "
    "and but if or because as until while of at by for with about against "
    "between into through during before after above below to from up down "
    "in out on off over under again further then once here there when where "
    "why how all both each few more most other some such no nor not only "
    "own same so than too very s t can will just don should now d ll m o "
    "re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn "
    "needn shan shouldn wasn weren won wouldn".split()
)

# Simple suffix-stripping lemmatiser (covers most common English inflections)
def _lemmatize(word: str) -> str:
    for suffix, replacement in [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("ising", "ise"),
        ("izing", "ize"), ("ised", "ise"), ("ized", "ize"),
        ("ational", "ate"), ("fulness", "ful"), ("ousness", "ous"),
        ("iveness", "ive"), ("nesses", ""), ("ations", "ate"),
        ("ation", "ate"), ("ating", "ate"), ("ated", "ate"),
        ("ings", ""), ("ing", ""), ("ness", ""), ("ment", ""),
        ("ful", ""), ("ous", ""), ("ive", ""), ("ies", "y"),
        ("ied", "y"), ("ed", ""), ("ly", ""), ("er", ""),
        ("est", ""), ("s", ""),
    ]:
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[: len(word) - len(suffix)] + replacement
    return word


def clean_text(text: str) -> str:
    """Full NLP preprocessing pipeline for a raw text string."""
    text = re.sub(r"<.*?>", "", text)           # strip HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)   # keep letters only
    tokens = text.lower().split()
    tokens = [
        _lemmatize(t)
        for t in tokens
        if t not in STOPS and len(t) > 2
    ]
    return " ".join(tokens)
