"""Verify all Project 3 Word2Vec artifacts against the frozen 3,000 corpus.

Run after training: python src/e3_03_verify_word2vec_3000.py
"""
from m01_corpus_reader import read_corpus, sha256
from project3_config import CORPUS_PATH, EMBEDDING_MODELS, FORMAL_DOCUMENT_COUNT, FORMAL_CORPUS_FILE_SHA256
from e3_01_word2vec_viewer import read_training_report, load_trained_word2vec, compare_embeddings


def main():
    assert sha256(CORPUS_PATH) == FORMAL_CORPUS_FILE_SHA256
    docs, header = read_corpus()
    assert len(docs) == FORMAL_DOCUMENT_COUNT
    report = read_training_report(expected_corpus_sha256=header["corpus_sha256"])
    assert report["corpus_stats"]["documents"] == FORMAL_DOCUMENT_COUNT
    for label in EMBEDDING_MODELS:
        model = load_trained_word2vec(label)
        assert model.wv.vector_size == 100
        assert len(model.wv) > 0
        probe = next(iter(model.wv.key_to_index))
        result = compare_embeddings(model, probe, top_k=3)
        assert result["term_exists"] and len(result["neighbors"]) == 3
        print(f"PASS: {label}, vocab={len(model.wv):,}, probe={probe}")
    print("PASS: all three Word2Vec models match Project 3 formal 3,000 corpus")


if __name__ == "__main__":
    main()
