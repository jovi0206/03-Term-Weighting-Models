"""Train Project 3 Word2Vec on the SAME 3,000 JATS Main Abstracts as VSM.

Usage (from Project 3 root):
    python src/e3_02_train_word2vec_3000.py
    python src/e3_02_train_word2vec_3000.py --force

The retrieval index uses bag-of-words preprocessing C. Word2Vec instead splits
abstracts into sentences to preserve local context, then applies the SAME C
word tokenizer. Window/context is thus meaningful; not a document retriever.
Never touches Project 2 or Project 3's frozen 1,000-record backup.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from m01_corpus_reader import read_corpus, sha256
from m02_text_processor import tokens
from project3_config import (CORPUS_PATH, EMBEDDING_DIR, EMBEDDING_MODELS,
    EMBEDDING_REPORT_PATH, FORMAL_DOCUMENT_COUNT, FORMAL_CORPUS_FILE_SHA256)

# A conservative sentence splitter for offline context construction; it does
# not pretend to replicate Project 2 M03/pySBD sentence boundary behavior.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
SETUPS = {
    "cbow_w5": {"sg": 0, "window": 5},
    "skipgram_w5": {"sg": 1, "window": 5},
    "skipgram_w2": {"sg": 1, "window": 2},
}


def collect_sentences(documents):
    sentences = []
    frequencies = Counter()
    per_doc_with_tokens = 0
    for document in documents:
        found = False
        for raw_sentence in SENTENCE_BOUNDARY.split(document["abstract"]):
            words = tokens(raw_sentence, "C")
            if len(words) < 2:
                continue
            sentences.append(words)
            frequencies.update(words)
            found = True
        if found:
            per_doc_with_tokens += 1
    if not sentences or per_doc_with_tokens != FORMAL_DOCUMENT_COUNT:
        raise ValueError("Unexpected empty training documents/sentences")
    return sentences, frequencies


def run(epochs=20, force=False):
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(f"Formal 3,000-doc corpus missing: {CORPUS_PATH}")
    file_digest = sha256(CORPUS_PATH)
    if file_digest != FORMAL_CORPUS_FILE_SHA256:
        raise ValueError(f"Unexpected 3,000 corpus file SHA: {file_digest}")
    documents, header = read_corpus()
    if len(documents) != FORMAL_DOCUMENT_COUNT:
        raise ValueError(f"Expected {FORMAL_DOCUMENT_COUNT} docs, got {len(documents)}")
    if (not header.get("corpus_sha256") or
            len({doc["pmid"] for doc in documents}) != FORMAL_DOCUMENT_COUNT):
        raise ValueError("Corpus has invalid document identifiers/provenance")
    if EMBEDDING_REPORT_PATH.is_file() and not force:
        from e3_01_word2vec_viewer import read_training_report, model_inventory
        try:
            old = read_training_report(expected_corpus_sha256=header["corpus_sha256"])
            if old.get("params", {}).get("epochs") != epochs or not all(model_inventory().values()):
                raise ValueError("Training parameters or model files differ")
        except (ValueError, OSError, FileNotFoundError) as exc:
            raise RuntimeError("Existing Project 3 models incompatible; use --force to retrain") from exc
        print("PASS: Project 3 Word2Vec models already match 3,000 corpus; no retraining")
        return old

    print(f"Source: {CORPUS_PATH} ({len(documents):,} Main Abstracts)")
    sentences, counts = collect_sentences(documents)
    print(f"Preprocessing C: {len(sentences):,} sentences; {sum(counts.values()):,} tokens")
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError("Install gensim: python -m pip install -r requirements.txt") from exc
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "corpus_id": "glp1_3000", "corpus_sha256": header["corpus_sha256"],
        "corpus_file_sha256": file_digest,
        "corpus_stats": {"documents": len(documents), "sentences": len(sentences),
                         "tokens": sum(counts.values()), "unique_tokens": len(counts),
                         "abstract_scope": "main_abstract_body_only"},
        "source": "Project 3 formal 3,000-document PMC JATS Main Abstract Corpus",
        "tokenization": "Project 3 preprocessing C; regex sentence boundaries; min 2 words/sentence",
        "params": {"vector_size": 100, "min_count": 2, "epochs": epochs,
                   "workers": 1, "seed": 42, "negative": 5, "sample": 0.001},
        "experiments": {}, "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with TemporaryDirectory(prefix="p3w2v_", dir=EMBEDDING_DIR) as tmp:
        tmp_dir = Path(tmp)
        for label, filename in EMBEDDING_MODELS.items():
            stem = filename.removesuffix(".model")
            setup = SETUPS[stem]
            print(f"Training {label}: window={setup['window']}, epochs={epochs} …", flush=True)
            model = Word2Vec(sentences=sentences, vector_size=100,
                window=setup["window"], min_count=2, workers=1,
                sg=setup["sg"], seed=42, epochs=epochs,
                negative=5, sample=0.001)
            model_file = tmp_dir / filename
            # Force a self-contained Gensim archive, no .npy sidecars to orphan.
            model.save(str(model_file), sep_limit=2**31)
            report["experiments"][stem] = {
                "params": {**report["params"], "window": setup["window"],
                           "architecture": "CBOW" if setup["sg"] == 0 else "Skip-gram"},
                "vocabulary_size": len(model.wv),
                "trained_on_sentences": len(sentences),
                "model_file": filename,
                "model_sha256": sha256(model_file),
            }
            if len(model.wv) == 0:
                raise RuntimeError(f"Empty Word2Vec vocabulary: {stem}")
            print(f"[OK] {label} · vocab={len(model.wv):,}", flush=True)
            del model
        report_file = tmp_dir / EMBEDDING_REPORT_PATH.name
        with gzip.open(report_file, "wt", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # Publish the metadata only after ALL models were atomically replaced.
        # Viewers validate SHA against the model files, so interrupted commits
        # cannot accidentally treat mixed versions as one coherent experiment.
        for name in EMBEDDING_MODELS.values():
            os.replace(tmp_dir / name, EMBEDDING_DIR / name)
        os.replace(report_file, EMBEDDING_REPORT_PATH)
    from e3_01_word2vec_viewer import read_training_report
    read_training_report(expected_corpus_sha256=header["corpus_sha256"])
    print("PASS: 3 Project 3 Word2Vec models trained on formal 3,000 documents")
    print("Model folder:", EMBEDDING_DIR)
    print("Report:", EMBEDDING_REPORT_PATH)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20, help="Default 20; override for local experimentation")
    parser.add_argument("--force", action="store_true", help="Retrain 3 models in Project 3 folder")
    args = parser.parse_args()
    run(epochs=args.epochs, force=args.force)


if __name__ == "__main__":
    main()
