"""
preprocessor.py — Text cleaning and NLP preprocessing pipeline.
Uses NLTK + SpaCy for tokenisation, stopword removal, and lemmatisation.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache
from typing import Optional

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
import nltk

def _download_nltk():
    for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

_download_nltk()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── SpaCy (optional — graceful fallback) ─────────────────────────────────────
_nlp = None

def _load_spacy():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    except Exception:
        _nlp = None
    return _nlp


# ── Constants ─────────────────────────────────────────────────────────────────
_STOP_WORDS = set(stopwords.words("english")) - {
    "not", "no", "nor", "but", "however", "what", "how", "why",
    "when", "where", "who", "which", "can", "could", "would",
}
_LEMMATIZER = WordNetLemmatizer()

_CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "can't": "cannot", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "that's": "that is",
    "what's": "what is", "where's": "where is", "there's": "there is",
}


def expand_contractions(text: str) -> str:
    tokens = text.lower().split()
    return " ".join(_CONTRACTIONS.get(t, t) for t in tokens)


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, expand contractions, remove noise."""
    text = text.lower().strip()
    text = expand_contractions(text)
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # non-alphanumeric
    text = re.sub(r"\s+", " ", text).strip()             # whitespace
    return text


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenise → lemmatise → (optionally) remove stopwords."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    lemmas = [_LEMMATIZER.lemmatize(t) for t in tokens if t.isalpha()]
    if remove_stopwords:
        lemmas = [t for t in lemmas if t not in _STOP_WORDS]
    return lemmas


def preprocess(text: str, remove_stopwords: bool = True) -> str:
    """Full pipeline → cleaned string suitable for TF-IDF or embedding."""
    return " ".join(tokenize_and_lemmatize(text, remove_stopwords))


def extract_entities_spacy(text: str) -> list[dict]:
    """Named entity recognition via SpaCy (returns empty list if unavailable)."""
    nlp = _load_spacy()
    if nlp is None:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


def detect_language_hint(text: str) -> str:
    """Very lightweight language hint — returns 'en' always for now.
    Swap with langdetect in production for multilingual support."""
    return "en"


@lru_cache(maxsize=1024)
def cached_preprocess(text: str) -> str:
    return preprocess(text)
