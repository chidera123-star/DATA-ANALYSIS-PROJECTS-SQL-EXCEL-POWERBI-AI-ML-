"""
translate.py — Helsinki-NLP MarianMT translation module
CSC 309 Project #14 — Machine Translation System
"""

from transformers import MarianMTModel, MarianTokenizer
import streamlit as st

MODELS = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de",
}


@st.cache_resource(show_spinner=False)
def load_model(lang: str):
    """Load and cache the tokeniser + model for the given language code."""
    model_name = MODELS[lang]
    tokenizer  = MarianTokenizer.from_pretrained(model_name)
    model      = MarianMTModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def translate_text(
    text: str,
    lang: str,
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    num_beams: int = 5,
    max_length: int = 128,
) -> str:
    """Translate a single English sentence to the target language."""
    if not text.strip():
        return ""

    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    outputs = model.generate(
        **inputs,
        num_beams=num_beams,
        max_length=max_length,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
