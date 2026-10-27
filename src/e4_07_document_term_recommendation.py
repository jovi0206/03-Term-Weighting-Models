"""E4-07: Explainable, document-grounded A5 Term B recommendations.

PRESENTATION/SELECTION ONLY: uses the already built M03 corpus counters,
DF/CF/IDF, the unchanged E4 weighting function, and M02 tokenizer.
No retrieval, weighting, index, or evaluation formula is changed.
The ranking is a lexical/statistical heuristic, NOT semantic understanding.
"""
from __future__ import annotations

import math
import re

from e4_03_term_weighting import tf_weight
from m02_text_processor import tokens

_VALID_TERM = re.compile(r"^[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*$", re.ASCII)
# Parser residues, headings and generic report scaffolding, not medical concepts.
_EXCLUDED = {
    'abstract', 'background', 'introduction', 'objective', 'objectives',
    'methods', 'method', 'results', 'result', 'conclusion', 'conclusions',
    'keywords', 'keyword', 'copyright', 'reserved', 'publisher', 'author',
    'authors', 'study', 'studies', 'review', 'reviews', 'analysis', 'analyses',
    'data', 'group', 'groups', 'using', 'used', 'associated', 'significant',
    'significantly', 'however', 'therefore', 'including', 'compared',
    'following', 'among', 'respectively', 'observed', 'reported',
    'table', 'figure', 'fig', 'doi', 'pmid', 'pmc', 'ncbi', 'et', 'al',
    'may', 'one', 'two', 'three', 'per', 'vs', 'www', 'http', 'https',
}
# Explicitly illustrative biomedical vocabulary; never described as a learned model.
_BIOMEDICAL_TERMS = {
    'semaglutide', 'liraglutide', 'tirzepatide', 'exenatide', 'dulaglutide',
    'retatrutide', 'insulin', 'glucagon', 'diabetes', 'diabetic', 'obesity',
    'obese', 'adiposity', 'glycemia', 'glycaemic', 'glycemic', 'glucose',
    'hba1c', 'hypoglycemia', 'hyperglycemia', 'cardiovascular', 'cardiac',
    'atherosclerosis', 'pancreatitis', 'pancreatic', 'gastric', 'pneumatosis',
    'kidney', 'renal', 'hepatic', 'liver', 'receptor', 'agonist', 'agonists',
    'metabolic', 'metabolism', 'cholesterol', 'triglycerides', 'lipid',
    'inflammation', 'mortality', 'hypertension', 'blood', 'pressure',
    'retinopathy', 'nephropathy', 'neuropathy', 'albuminuria',
    'placebo', 'randomized', 'randomised', 'therapy', 'treatment',
    'dose', 'dosing', 'safety', 'efficacy', 'adverse', 'risk',
}
_BIO_PREFIX = ('cardio', 'neuro', 'nephro', 'pancrea', 'gluco', 'glyc',
               'hepato', 'gastro', 'metabol', 'insulin', 'pharma', 'atheroscler')
_BIO_SUFFIX = ('itis', 'emia', 'pathy', 'glycemia', 'glycaemia',
               'glutide', 'receptor', 'inhibitor', 'agonist', 'peptide')


def _medical_relevance(term: str) -> float:
    if term in _BIOMEDICAL_TERMS:
        return 2.0
    if term.startswith(_BIO_PREFIX) or term.endswith(_BIO_SUFFIX):
        return 1.0
    if ('-' in term and any(part in ('glp', 'gip', 'sglt', 'dpp', 't2d')
                            for part in term.split('-'))):
        return 1.0
    return 0.0


def recommend_document_terms(index, document_index: int, term_a: str = 'glp-1', limit: int = 5):
    """Rank <=5 verified terms in a *selected* Main Abstract after C.

    Scoring: title match + explicit biomedical term rules + IDF greater than
    baseline + actual document Log TF-IDF, with a small Document TF tie-break.
    Every return row carries auditable TF/CF/DF/IDF. Never fills absent words.
    """
    if not 0 <= document_index < len(index.docs):
        raise IndexError('Document index out of range')
    if limit < 1:
        raise ValueError('limit must be positive')
    counts = index.counters[document_index]
    title_terms = set(tokens(index.docs[document_index].get('title', ''), index.condition))
    baseline = tokens(term_a, index.condition)
    baseline_term = baseline[0] if len(baseline) == 1 else term_a.casefold().strip()
    baseline_col = index.vocabulary.get(baseline_term)
    baseline_idf = (float(index.idfs[baseline_col]) if baseline_col is not None else None)
    candidates = []
    for term, frequency in counts.items():
        if (term == baseline_term or not _VALID_TERM.fullmatch(term)
                or len(term) < 3 or term in _EXCLUDED or frequency < 1):
            continue
        col = index.vocabulary.get(term)
        if col is None:
            continue
        cf = int(index.cf.get(term, 0))
        df = int(index.df.get(term, 0))
        term_idf = float(index.idfs[col])
        if df < 1 or cf < frequency or not math.isfinite(term_idf):
            continue
        in_title = term in title_terms
        biomedical = _medical_relevance(term)
        contrast = baseline_idf is not None and term_idf > baseline_idf
        log_tfidf = float(tf_weight(int(frequency), 'Log TF-IDF', term_idf))
        # Transparent heuristic; term meaning is NOT inferred by AI.
        score = (2.5 * in_title + 1.4 * biomedical + 1.6 * contrast
                 + math.log1p(log_tfidf) + .06 * min(int(frequency), 20))
        candidates.append({
            'Term': term, 'Document TF': int(frequency), 'Corpus CF': cf,
            'Corpus DF': df, 'IDF (log10)': term_idf,
            'Higher IDF than Term A': bool(contrast),
            'In title': bool(in_title), 'Medical rule': biomedical > 0,
            'Recommendation score': round(score, 6),
        })
    candidates.sort(key=lambda row: (-row['Recommendation score'],
                                      -row['IDF (log10)'], -row['Document TF'],
                                      row['Term']))
    return candidates[:limit]


def diagnose_document_term(index, document_index: int, raw_term: str) -> dict:
    """Determine why an entered term has zero *indexed abstract* TF.

    Distinguishes corpus-wide OOV, title-only, removed/changed by condition C,
    and simply absent from the selected Main Abstract. Diagnostics are read-only.
    """
    if not 0 <= document_index < len(index.docs):
        raise IndexError('Document index out of range')
    canonical = tokens(raw_term, index.condition)
    if len(canonical) != 1:
        return {'normalized': canonical, 'reason': 'Input does not normalize to exactly one term.'}
    term = canonical[0]
    doc = index.docs[document_index]
    tf = int(index.counters[document_index].get(term, 0))
    if tf > 0:
        reason = 'Present in indexed Main Abstract.'
    elif term not in index.vocabulary:
        reason = 'Not in the official corpus vocabulary (OOV).'
    else:
        # B differs from C in stopword filtering; exact raw input matching may
        # be lost due to case/hyphen normalization. B helps isolate that case.
        raw_b = tokens(doc.get('abstract', ''), 'B')
        if term in raw_b:
            reason = 'Appears with preprocessing B but not after the selected preprocessing condition.'
        elif term in tokens(doc.get('title', ''), index.condition):
            reason = 'Appears in the TITLE, but the retrieval index covers Main Abstract only.'
        else:
            reason = 'Not present as this token in the selected Main Abstract after preprocessing.'
    return {
        'pmid': str(doc['pmid']), 'normalized': term, 'document_tf': tf,
        'corpus_df': int(index.df.get(term, 0)), 'reason': reason,
    }


def term_b_on_document_change(previous_context, new_context, current_term_b,
                              user_edited, first_recommended):
    """State policy: automatic B follows document/Term A until manually edited."""
    if previous_context != new_context and not user_edited:
        return first_recommended
    return current_term_b
