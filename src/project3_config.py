"""Project 3 shared configuration. No processing or UI imports here.

The current assignment owns its module plan. Earlier Project 2 numbering is
not a contract imposed on later assignments.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "prepared_data" / "glp1_3000_abstracts.jsonl.gz"
FORMAL_DOCUMENT_COUNT = 3000
# Immutable official corpus produced by the user's Batch 2 run, not the previous 1K data.
FORMAL_CORPUS_FILE_SHA256 = "c67cef4af612bab3a32ca6b1c13229a74a965a4e29d823a7118c37dcb171ede6"
RULES_PATH = ROOT / "data" / "query_relevance_rules.json"
RESULTS_PATH = ROOT / "results"

METRICS = ("Cosine", "Raw Dot Product", "L2 Dot Product")
CONDITIONS = {
    "A": "Basic: whitespace + casefold",
    "B": "Remove punctuation: M02 tokenizer + casefold",
    "C": "B + stopword removal (formal baseline)",
    "D": "C + Project 2 Porter stemming",
}
FORMAL_CONDITION = "C"
DEFAULT_QUERY = "semaglutide weight loss"

# Project 3 optional word-vector comparison. Train NEW models from the same formal
# 3,000 Main Abstracts. No Project 2 models are ever loaded at runtime.
EMBEDDING_DIR = ROOT / "artifacts" / "embedding_experiments" / "glp1_3000"
EMBEDDING_REPORT_PATH = EMBEDDING_DIR / "embedding_report.json.gz"
EMBEDDING_MODELS = {
    "CBOW W5": "cbow_w5.model",
    "Skip-gram W5": "skipgram_w5.model",
    "Skip-gram W2": "skipgram_w2.model",
}
