"""Offline regression: synthetic corpus and FAKE UMAP backend, not a 3k fit.

A real umap-learn 3k fit must be run on the user's Windows environment.
This suite deliberately never fabricates rank/score outputs or claims a real fit.
"""
import ast
import gzip
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from e4_03_term_weighting import WEIGHTINGS
from e4_06_weighting_retrieval_comparison import compare_five_method_retrieval
from e8_02_shared_umap import (REFERENCE_WEIGHTING, _normalized_reference_docs,
    fit_shared_umap, project_shared_query, plot_inputs)
from m08_controller import SearchIndex
from project3_charts import build_shared_umap_ranking_chart, shared_umap_axis_ranges


class FakeUMAP:
    """Only supports deterministic plumbing tests, never used by the real app."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def fit_transform(self, sparse_docs):
        assert sparse_docs.getformat() == 'csr'
        assert sparse_docs.dtype == np.float32
        self.fitted = sparse_docs
        # For test purposes, stable coordinates uniquely identify document rows.
        return np.column_stack((np.arange(sparse_docs.shape[0], dtype=float),
                                np.arange(sparse_docs.shape[0], dtype=float) * .5))
    def transform(self, sparse_query):
        assert sparse_query.shape[0] == 1
        assert sparse_query.getformat() == 'csr'
        return np.array([[3.5, 2.5]], dtype=float)


class TestSharedUMAP(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        path = Path(cls.tmp.name) / 'synthetic.jsonl.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as file:
            file.write(json.dumps({'record_type': 'project2_corpus_header',
                                   'corpus_sha256': 'synthetic'}) + '\n')
            for i in range(16):
                abstract = ('semaglutide ' * (1 + i % 3) +
                            'weight ' * (1 + i % 4) +
                            'loss ' * (i % 3) +
                            'diabetes ' * (i % 5 + 1) +
                            f'uniquetoken{i} ' * (i % 2 + 1))
                doc = {'pmid': str(80000001 + i), 'pmcid': f'PMC{90001+i}',
                       'title': f'Synthetic GLP1 study {i}', 'abstract_clean': abstract}
                file.write(json.dumps(doc) + '\n')
        cls.index = SearchIndex.build(corpus_path=path, condition='C')
        cls.query = 'semaglutide weight loss'
        cls.a1 = compare_five_method_retrieval(cls.index, cls.query)
        with mock.patch.dict(sys.modules, {'umap': SimpleNamespace(UMAP=FakeUMAP)}):
            cls.projection = fit_shared_umap(cls.index)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reference_only_log_tfidf_and_sparse_l2_normalized(self):
        self.assertEqual(REFERENCE_WEIGHTING, 'Log TF-IDF')
        valid, matrix = _normalized_reference_docs(self.index)
        self.assertEqual(matrix.shape, (len(valid), len(self.index.vocabulary)))
        self.assertTrue(np.allclose(np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()), 1))
        self.assertEqual(self.projection.model.kwargs['metric'], 'cosine')
        self.assertEqual(self.projection.model.kwargs['random_state'], 42)
        self.assertEqual(self.projection.parameters['corpus_sha256'], self.index.corpus_sha256)
        self.assertEqual(self.projection.parameters['preprocessing'], 'C')

    def test_one_reference_map_over_all_methods(self):
        coords = self.projection.coords
        positions = self.projection.valid_indices
        self.assertEqual(coords.shape, (len(self.index.docs), 2))
        self.assertEqual(len(positions), 16)
        for method in WEIGHTINGS:
            result = self.a1['rankings'][method]
            plot = plot_inputs(self.index, self.projection, method,
                               result, self.a1['labels'], query_xy=(3.5, 2.5))
            expected_pmids = [str(hit['pmid']) for hit in result['hits']]
            self.assertEqual([h['pmid'] for h in plot['top']], expected_pmids)
            self.assertEqual([h['score'] for h in plot['top']], [hit['score'] for hit in result['hits']])
            for hit in plot['top']:
                self.assertTrue(np.allclose(coords[hit['doc_index']],
                                            coords[self.a1['pmid_to_index'][hit['pmid']]]))
            self.assertEqual(len(plot['other_indices']) + len(plot['top']), 16)

    def test_query_is_transformed_with_same_model_not_refit(self):
        xy, warning = project_shared_query(self.index, self.projection, self.query)
        self.assertEqual(xy, (3.5, 2.5))
        self.assertIsNone(warning)
        no_xy, warning = project_shared_query(self.index, self.projection, 'unknowntopiczzz')
        self.assertIsNone(no_xy)
        self.assertIn('omitted', warning)
        # FakeUMAP has one fit_transform and transform; no fit called during query.
        self.assertTrue(hasattr(self.projection.model, 'fitted'))

    def test_chart_style_hover_score_fixed_axes(self):
        axes = shared_umap_axis_ranges(np.vstack((self.projection.coords, [3.5, 2.5])))
        for method in WEIGHTINGS:
            figure = build_shared_umap_ranking_chart(
                self.index, self.projection, method, self.a1['rankings'][method],
                self.a1['labels'], (3.5, 2.5), axes)
            self.assertEqual(len(figure.data), 3)
            self.assertEqual(figure.data[0].name, 'Other documents')
            self.assertEqual(figure.data[1].name, 'Top 10 (ranks 1–3 labeled)')
            self.assertEqual(figure.data[2].marker.symbol, 'star')
            self.assertEqual(figure.data[2].marker.color, '#C83B3B')
            self.assertEqual(figure.data[1].marker.color, '#E88929')
            self.assertEqual(list(figure.layout.xaxis.range), axes[0])
            self.assertEqual(list(figure.layout.yaxis.range), axes[1])
            self.assertTrue(all('Original Cosine' in txt for txt in figure.data[1].hovertext))
            self.assertEqual(list(figure.data[1].text[:3]), ['1', '2', '3'])
            self.assertEqual(len(figure.data[1].x), len(self.a1['rankings'][method]['hits']))

    def test_reject_wrong_model_method_hit_identity(self):
        result = dict(self.a1['rankings']['Raw TF'])
        result['metric'] = 'Raw Dot Product'
        with self.assertRaises(ValueError):
            plot_inputs(self.index, self.projection, 'Raw TF', result, self.a1['labels'])
        with self.assertRaises(ValueError):
            plot_inputs(self.index, self.projection, 'Wrong TF', self.a1['rankings']['Raw TF'], self.a1['labels'])
        result = dict(self.a1['rankings']['Raw TF'])
        bad_hit = dict(result['hits'][0])
        bad_hit['pmid'] = 'NOT-A-REAL-PMID'
        result['hits'] = [bad_hit]
        with self.assertRaises(ValueError):
            plot_inputs(self.index, self.projection, 'Raw TF', result, self.a1['labels'])

    def test_top1_agreement_is_frequency_not_boolean(self):
        summary = self.a1['summary']
        counts = {}
        for method, pmid in summary['top1_by_method'].items():
            if pmid is not None:
                counts[pmid] = counts.get(pmid, 0) + 1
        self.assertEqual(summary['top1_max_agreement'], max(counts.values(), default=0))
        self.assertEqual(summary['top1_unique_pmids'], len(counts))
        for group in summary['top1_agreement_groups']:
            self.assertEqual(len(group['methods']), summary['top1_max_agreement'])
            self.assertTrue(all(summary['top1_by_method'][m] == group['pmid']
                                for m in group['methods']))

    def test_top1_agreement_two_of_five_and_tied_groups(self):
        class FakeIndex:
            def __init__(self, picks):
                self.picks = picks
            def search(self, query, method, metric, limit):
                pmid = self.picks[method]
                return {"method": method, "query": query, "metric": metric,
                        "hits": [{"rank": 1, "pmid": pmid, "doc_index": int(pmid),
                                  "title": "Synthetic", "score": 0.5}]}
        # One pair shares its first PMID; three other methods choose distinct docs.
        picks = dict(zip(WEIGHTINGS, ['1', '1', '2', '3', '4']))
        summary = compare_five_method_retrieval(FakeIndex(picks), 'q')['summary']
        self.assertEqual(summary['top1_max_agreement'], 2)
        self.assertEqual(summary['top1_unique_pmids'], 4)
        self.assertEqual(summary['top1_agreement_groups'],
                         [{'pmid': '1', 'methods': list(WEIGHTINGS[:2])}])
        # Two equally common Top-1 pairs must both be disclosed.
        picks = dict(zip(WEIGHTINGS, ['1', '1', '2', '2', '3']))
        summary = compare_five_method_retrieval(FakeIndex(picks), 'q')['summary']
        self.assertEqual(summary['top1_max_agreement'], 2)
        self.assertEqual([g['pmid'] for g in summary['top1_agreement_groups']], ['1', '2'])
        self.assertEqual(summary['top1_unique_pmids'], 3)

    def test_oov_query_does_not_generate_artificial_results(self):
        output = compare_five_method_retrieval(self.index, 'unknownwordzzzz')
        self.assertEqual(output['summary']['top1_max_agreement'], 0)
        self.assertEqual(output['summary']['top1_unique_pmids'], 0)
        self.assertEqual(output['summary']['top1_agreement_groups'], [])
        self.assertEqual(output['summary']['unique_top10_pmids'], 0)

    def test_only_02_section_changed_and_06_remains(self):
        src = (SRC / 'm09_app.py').read_text(encoding='utf-8')
        tree = ast.parse(src)
        self.assertTrue(tree)
        part = src.split('elif page == "02 Weighting Comparison":')[1].split('elif page == "03 Controlled Experiments":')[0]
        for label in ('### A1 · Five Methods Retrieval Comparison',
                      '### A2 · Five Methods UMAP Projection',
                      '### A3 · Two Terms × Five Methods',
                      '#### A3 · Two terms'):
            self.assertIn(label, part)
        self.assertLess(part.index('### A1 ·'), part.index('### A2 ·'))
        self.assertLess(part.index('### A2 ·'), part.index('### A3 ·'))
        self.assertIn('cached_shared_umap(condition, index.corpus_sha256)', part)
        self.assertIn('wcmp_a1_select_', part)
        self.assertIn('Top 1 Agreement', part)
        self.assertNotIn('Same Top 1 in all five?', part)
        self.assertIn('Full Top 10', part)
        self.assertIn('"PMID"', part)

if __name__ == '__main__':
    import unittest
    unittest.main()
