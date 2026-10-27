"""M02: Tokenize, normalize and apply Project 3 preprocessing conditions.

Reuses Project 2 E21 Porter and E22 stopwords without importing its full
JATS parsing / sentence segmentation stack. WORD_PATTERN and normalization
were compared with the uploaded Project 2 M02 source.
"""
import re
from e2_01_porter import porter_stem
from e2_02_preprocessing_experiment import DEFAULT_ENGLISH_STOPWORDS
from e4_02_term_statistics import compute_idf
from project3_config import CONDITIONS

WORD_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*(?:[‐‑‒–—-]\d+(?:[.,]\d+)*)?(?:%|[A-Za-z]+)?"
    r"|"
    r"[^\W_]+"
    r"(?:[.'’/‐‑‒–—-][^\W_]+)*"
    r"%?",
    re.UNICODE,
)


def normalize_text(text):
    if not text:
        return ""
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    return re.sub(r"([(\[])\s+", r"\1", text)


def tokenize_words(text):
    return WORD_PATTERN.findall(normalize_text(text))


def tokens(text, condition="C"):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "A":
        return [item.casefold() for item in re.findall(r"\S+", str(text or "")) if item.strip()]
    output = [item.casefold().strip() for item in tokenize_words(text) if item.strip()]
    if condition in ("C", "D"):
        output = [item for item in output if item not in DEFAULT_ENGLISH_STOPWORDS]
    if condition == "D":
        output = [stem for item in output if (stem := porter_stem(item))]
    return output


def idf(n_documents, document_frequency):
    """Delegate to unchanged P2 E42: unsmoothed log10(N / DF)."""
    return compute_idf(n_documents, document_frequency, log_base=10)
