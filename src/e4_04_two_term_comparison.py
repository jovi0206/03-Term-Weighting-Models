"""E4: Explain two term weights in ONE document using the existing index.

The TF/IDF formulas and corpus index are not recomputed. Reuses the same
M02 preprocessing and E4 policies as retrieval so comparisons are faithful.
"""
from m02_text_processor import tokens
from e4_03_term_weighting import WEIGHTINGS, tf_weight


def compare_two_terms(index, term_a: str, term_b: str, document_index: int):
    """Return two rows with corpus frequency and within-document weights.

    Raises ValueError if a field does not normalize to exactly one token or
    both fields normalize to the same token. An out-of-vocabulary single token
    is shown with DF=0, IDF=None, and all document weights zero.
    """
    if not 0 <= document_index < len(index.docs):
        raise IndexError("Document index out of range")
    normalized = []
    for raw in (term_a, term_b):
        items = tokens(raw, index.condition)
        if len(items) != 1:
            raise ValueError(f"Please enter exactly one valid token per field: {raw!r} → {items}")
        normalized.append(items[0])
    if normalized[0] == normalized[1]:
        raise ValueError("Both inputs normalize to the same token; choose two different terms")

    doc_counts = index.counters[document_index]
    rows = []
    for entered, term in zip((term_a, term_b), normalized):
        tf = int(doc_counts.get(term, 0))
        present = term in index.vocabulary
        idf_value = float(index.idfs[index.vocabulary[term]]) if present else None
        values = {method: tf_weight(tf, method, idf_value or 0.0) for method in WEIGHTINGS}
        rows.append({
            "Input": entered.strip(),
            "Term": term,
            "Doc TF": tf,
            "Corpus CF": int(index.cf.get(term, 0)),
            "Corpus DF": int(index.df.get(term, 0)),
            "IDF (log10)": idf_value,
            **values,
        })
    return rows
