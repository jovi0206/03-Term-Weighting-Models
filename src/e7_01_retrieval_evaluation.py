"""Deterministic, transparently rule-labeled relevance proxy and IR metrics.

IMPORTANT: Source corpus has NO independently judged relevance labels. The
provided judgments are topic-pattern proxies (not manually verified qrels).
They are used for teaching and controlled comparisons, NOT ground truth.
"""
from pathlib import Path
import json
import math
import re

from project3_config import RULES_PATH
from e4_03_term_weighting import WEIGHTINGS


def load_rules(path=RULES_PATH):
    rules = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(rules) < 10 or len({r['id'] for r in rules}) != len(rules):
        raise ValueError("At least 10 distinct relevance rules are required")
    for r in rules:
        if not r.get("all_concepts") or not r.get("query"):
            raise ValueError(f"Invalid rule: {r.get('id')}")
    return rules


def rule_matches(text, rule):
    """AND across concepts, OR across each concept's regex variants.

    Regex labels use title? NO. Only the same main abstract body, like retrieval.
    This means judgments have keyword overlap with retrieval; see limitations.
    """
    return all(any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in variants)
               for variants in rule['all_concepts'])


def make_qrels(index, rules=None):
    rules = load_rules() if rules is None else rules
    return {rule['id']: {i for i, d in enumerate(index.docs)
                          if rule_matches(d['abstract'], rule)} for rule in rules}


def evaluate_ranking(hits, relevant, k=10):
    """Binary-relevance metrics, P@k denominator k; empty qrels -> undefined."""
    if not relevant:
        return {"Precision@10": None, "Recall@10": None, "MRR@10": None, "nDCG@10": None,
                "Relevant Count": 0, "Relevant Retrieved": 0}
    ids = [h['doc_index'] for h in hits[:k]]
    flags = [int(i in relevant) for i in ids]
    tp = sum(flags)
    p = tp / k
    recall = tp / len(relevant)
    rr = next((1.0 / j for j, flag in enumerate(flags, start=1) if flag), 0.0)
    dcg = sum(flag / math.log2(j + 1) for j, flag in enumerate(flags, start=1))
    ideal = sum(1 / math.log2(j+1) for j in range(1, min(k, len(relevant)) + 1))
    return {"Precision@10": p, "Recall@10": recall, "MRR@10": rr,
            "nDCG@10": dcg / ideal if ideal else 0.0,
            "Relevant Count": len(relevant), "Relevant Retrieved": tp}


def evaluate_all(index, rules=None, k=10):
    """All methods on the EXACT same fixed corpus/preprocessing/qrels/queries."""
    rules = load_rules() if rules is None else rules
    if len(rules) < 10:
        raise ValueError("Requires >=10 queries")
    qrels = make_qrels(index, rules)
    if any(not rel for rel in qrels.values()):
        empty = [rid for rid, rel in qrels.items() if not rel]
        raise ValueError(f"No proxy-relevant documents for queries {empty}")
    rows = []
    for method in WEIGHTINGS:
        for rule in rules:
            r = index.search(rule['query'], method, "Cosine", k)
            metrics = evaluate_ranking(r['hits'], qrels[rule['id']], k)
            rows.append({"Query ID": rule['id'], "Query": rule['query'], "Weighting": method,
                         **metrics, "Retrieved": len(r['hits']),
                         "Relevant PMIDs": "|".join(index.docs[i]['pmid'] for i in sorted(qrels[rule['id']]))})
    summary = []
    for method in WEIGHTINGS:
        sub = [r for r in rows if r['Weighting'] == method]
        entry = {"Weighting": method, "Queries": len(sub)}
        for m in ("Precision@10", "Recall@10", "MRR@10", "nDCG@10"):
            vals = [r[m] for r in sub if r[m] is not None]
            entry[m] = sum(vals)/len(vals) if vals else None
        summary.append(entry)
    return rows, summary, qrels
