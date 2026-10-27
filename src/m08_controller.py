"""M08: Small stable facade used by Streamlit, CLI and evaluation extensions.

All algorithm logic stays in M01–M07 and E4. M08 is an orchestration layer,
not a second implementation of tokenization, weighting or search.
"""
from project3_contracts import IndexData
from m03_index_builder import build_index
from m04_query_processor import query_vector
from m06_retrieval_engine import search
from m07_result_processor import explain_terms, term_stats, summary


class SearchIndex(IndexData):
    @classmethod
    def build(cls, corpus_path=None, condition="C"):
        return cls(**vars(build_index(corpus_path, condition)))

    def query_vector(self, query, method):
        return query_vector(self, query, method)

    def search(self, query, method="Raw TF-IDF", metric="Cosine", limit=10):
        return search(self, query, method, metric, limit)

    def explain_terms(self, query, document_index=None, method="Raw TF-IDF"):
        return explain_terms(self, query, document_index, method)

    def term_stats(self, term):
        return term_stats(self, term)

    def summary(self):
        return summary(self)
