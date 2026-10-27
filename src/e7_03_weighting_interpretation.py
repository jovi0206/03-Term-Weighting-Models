"""E7: Summarize actual evaluation evidence without changing any scores.

Reports method ranks *within the current experiment*, not universal algorithm
rankings. A tied score shares a rank; qrels remain proxy judgments.
"""
from e4_03_term_weighting import WEIGHTINGS


def method_performance_summary(means):
    """Summarize P@10 / nDCG@10 and competition ranks for five methods."""
    by_method = {row["Weighting"]: row for row in means}
    if len(means) != len(WEIGHTINGS) or set(by_method) != set(WEIGHTINGS):
        raise ValueError("Expected one evaluation row for each weighting method")
    output = []
    for method in WEIGHTINGS:
        row = by_method[method]
        score = float(row["nDCG@10"])
        rank = 1 + sum(float(other["nDCG@10"]) > score + 1e-12 for other in means)
        output.append({
            "Method": method,
            "P@10": float(row["Precision@10"]),
            "nDCG@10": score,
            "nDCG rank (of 5)": rank,
        })
    return output
