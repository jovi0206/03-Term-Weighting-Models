"""M07: Produce explainable term evidence, per-term stats and corpus summary.

Data shaping only. No Streamlit dependency; usable by CLI tests and later UI.
"""
from collections import Counter
from e4_03_term_weighting import tf_weight
from m02_text_processor import tokens


def explain_terms(index, query, document_index=None, method="Raw TF-IDF"):
    query_counts = Counter(tokens(query, index.condition))
    doc_counts = index.counters[document_index] if document_index is not None else Counter()
    entries = []
    for term in sorted(set(query_counts) | set(doc_counts),
                       key=lambda item: (-query_counts[item], -doc_counts[item], item)):
        if term not in index.vocabulary:
            continue
        col = index.vocabulary[term]
        idf = float(index.idfs[col])
        entries.append({
            "Term": term,
            "Query TF": query_counts[term],
            "Document TF": doc_counts[term],
            "CF": index.cf[term],
            "DF": index.df[term],
            "IDF (log10)": idf,
            "Query weight": tf_weight(query_counts[term], method, idf),
            "Document weight": tf_weight(doc_counts[term], method, idf),
        })
    return entries


def term_stats(index, term):
    output = []
    for token in tokens(term, index.condition):
        output.append({
            "Term": token,
            "CF": index.cf.get(token, 0),
            "DF": index.df.get(token, 0),
            "IDF (log10)": float(index.idfs[index.vocabulary[token]])
                if token in index.vocabulary else None,
            "Coverage": index.df.get(token, 0) / len(index.docs),
        })
    return output


def summary(index):
    n = len(index.docs)
    return {
        "documents": n,
        "unique_pmids": len({doc["pmid"] for doc in index.docs}),
        "vocabulary": len(index.vocabulary),
        "tokens": sum(index.cf.values()),
        "avg_terms_doc": sum(index.cf.values()) / n,
        "condition": index.condition,
        "corpus_sha256": index.corpus_sha256,
        "project2_header_sha256": index.header.get("corpus_sha256", ""),
    }
