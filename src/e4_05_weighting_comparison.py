"""E4: Consistent query/document term comparisons for Project 3 Batch 2.

Reuses M02 normalization, E4's existing 5 weighting policies, and E4's
existing two-term document comparison. No new TF/IDF implementation.
"""
from collections import Counter

from e4_03_term_weighting import WEIGHTINGS, tf_weight
from e4_04_two_term_comparison import compare_two_terms
from m02_text_processor import tokens


def compare_query_document_terms(index, query, document_index, terms):
    """Return (summary_rows, weight_rows) for one or two corpus-normalized terms.

    Each weight row is measured separately in a query and a single document.
    For a query term absent from the query, its Query TF and weight are zero.
    For a term absent from the corpus, IDF is None and all weights are zero.
    The index, weighting formulas and vocabulary are shared with retrieval.
    """
    if not 0 <= document_index < len(index.docs):
        raise IndexError("Document index out of range")
    if len(terms) not in (1, 2):
        raise ValueError("Compare exactly one or two terms")
    normalized = []
    for entered in terms:
        extracted = tokens(entered, index.condition)
        if len(extracted) != 1:
            raise ValueError(f"One token is required for each term: {entered!r} → {extracted}")
        normalized.append(extracted[0])
    if len(set(normalized)) != len(normalized):
        raise ValueError("Two fields become the same term after preprocessing")

    query_counts = Counter(tokens(query, index.condition))
    if len(terms) == 2:
        doc_rows = compare_two_terms(index, terms[0], terms[1], document_index)
    else:
        term = normalized[0]
        doc_tf = int(index.counters[document_index].get(term, 0))
        col = index.vocabulary.get(term)
        idfv = float(index.idfs[col]) if col is not None else None
        doc_rows = [{
            "Input": terms[0], "Term": term, "Doc TF": doc_tf,
            "Corpus CF": int(index.cf.get(term, 0)),
            "Corpus DF": int(index.df.get(term, 0)),
            "IDF (log10)": idfv,
            **{m: tf_weight(doc_tf, m, idfv or 0.0) for m in WEIGHTINGS},
        }]

    summaries, weights = [], []
    for doc_row in doc_rows:
        term = doc_row["Term"]
        qtf = int(query_counts.get(term, 0))
        dtf = doc_row["Doc TF"]
        idfv = doc_row["IDF (log10)"]
        summaries.append({
            "Term": term, "Query TF": qtf, "Document TF": dtf,
            "Corpus CF": doc_row["Corpus CF"],
            "Corpus DF": doc_row["Corpus DF"], "IDF (log10)": idfv,
        })
        for method in WEIGHTINGS:
            weights.append({
                "Term": term, "Weighting": method, "Query Weight": tf_weight(qtf, method, idfv or 0.0),
                "Document Weight": float(doc_row[method]),
            })
    return summaries, weights


def suggest_document_term_pair(index, document_index):
    """Suggest two actually present terms for a teaching demo, without overriding inputs.

    Favors semaglutide + GLP-1 if available. Otherwise prioritizes GLP-1
    topic vocabulary and picks a second term with contrasting TF / DF.
    The existing term weighting formulas and corpus frequency data are reused.
    """
    if not 0 <= document_index < len(index.docs):
        raise IndexError("Document index out of range")
    counts = index.counters[document_index]
    choices = [term for term in counts
               if term in index.vocabulary and len(term) >= 3
               and any(c.isalpha() for c in term)]
    if len(choices) < 2:
        return None
    keywords = ("semaglutide", "glp-1", "liraglutide", "obesity",
                "diabetes", "weight", "loss", "cardiovascular", "insulin")
    preferred = []
    for raw in keywords:
        normalized = tokens(raw, index.condition)
        if len(normalized) == 1 and normalized[0] in choices and normalized[0] not in preferred:
            preferred.append(normalized[0])
    if len(preferred) >= 2:
        pair = preferred[:2]
        if counts[pair[0]] != counts[pair[1]] or index.df[pair[0]] != index.df[pair[1]]:
            return tuple(pair)
    dominant = preferred[0] if preferred else max(choices, key=lambda t: (counts[t], -index.df[t]))
    pool = [term for term in choices if term != dominant]
    secondary = max(pool, key=lambda term: (
        bool(counts[term] != counts[dominant]),
        bool(index.df[term] != index.df[dominant]),
        abs(float(index.idfs[index.vocabulary[term]]) -
            float(index.idfs[index.vocabulary[dominant]])),
        counts[term], -index.df[term], term,
    ))
    return dominant, secondary
