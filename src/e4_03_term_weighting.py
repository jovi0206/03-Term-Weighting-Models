"""E4: Project 3 plug-in-ready term-weighting policies.

New weighting method: register its function in WEIGHTING_FUNCTIONS here; the
M03 index and M04 query use this same interface automatically. The registry
is intentionally tiny rather than a dynamic plugin framework.
"""
import math


def raw_tf(tf, idf_value):
    return float(tf)


def log_tf(tf, idf_value):
    return 1.0 + math.log10(tf)


def idf_only(tf, idf_value):
    return float(idf_value)


def raw_tfidf(tf, idf_value):
    return float(tf) * idf_value


def log_tfidf(tf, idf_value):
    return (1.0 + math.log10(tf)) * idf_value


WEIGHTING_FUNCTIONS = {
    "Raw TF": raw_tf,
    "Log TF": log_tf,
    "IDF Only": idf_only,
    "Raw TF-IDF": raw_tfidf,
    "Log TF-IDF": log_tfidf,
}


def tf_weight(tf, method, idf_value):
    if method not in WEIGHTING_FUNCTIONS:
        raise ValueError(f"Unknown weighting {method}")
    if tf <= 0:
        return 0.0
    return WEIGHTING_FUNCTIONS[method](tf, idf_value)


# Single source of truth for algorithms exposed in the UI, index and reports.
WEIGHTINGS = tuple(WEIGHTING_FUNCTIONS)
