"""02/A0 retrieval + selection regression; all docs here are SYNTHETIC, not formal 3k."""
import ast
import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

from e4_03_term_weighting import WEIGHTINGS
from e4_06_weighting_retrieval_comparison import (
    compare_five_method_retrieval, selected_pmid_after_query_change,
)
from e4_05_weighting_comparison import compare_query_document_terms
from m08_controller import SearchIndex
from project3_charts import (build_single_term_five_weight_chart,
    build_query_document_comparison_chart, build_two_term_five_weight_chart)


class TestFiveMethodRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = TemporaryDirectory()
        path = Path(cls.workspace.name) / 'synthetic.jsonl.gz'
        docs = []
        for i in range(15):
            # Intentional TF/DF contrasts, OOV terms, and varying document length.
            abstract = (
                ('semaglutide ' * (1 + i % 4))
                + ('weight ' * (1 + (i // 3) % 4))
                + ('loss ' * (i % 3))
                + ('diabetes ' * (i % 7 + 1))
                + ('cardiovascular ' * (i % 2))
                + (f'rarekeyword{i} ' * (1 + i % 2))
            ).strip()
            docs.append({'pmid': str(90000000 + i), 'pmcid': f'PMC{990000 + i}',
                         'title': f'Synthetic semaglutide paper {i}',
                         'abstract_clean': abstract})
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(json.dumps({'record_type': 'project2_corpus_header', 'corpus_sha256': 'synthetic'}) + '\n')
            for doc in docs:
                f.write(json.dumps(doc) + '\n')
        cls.index = SearchIndex.build(corpus_path=path, condition='C')
        cls.query = 'semaglutide weight loss'

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_five_methods_are_exact_m08_results(self):
        a0 = compare_five_method_retrieval(self.index, self.query)
        self.assertEqual(tuple(a0['rankings']), WEIGHTINGS)
        for method in WEIGHTINGS:
            expected = self.index.search(self.query, method, 'Cosine', limit=10)
            actual = a0['rankings'][method]
            self.assertEqual(actual['hits'], expected['hits'])
            self.assertEqual(actual['query'], self.query)
            self.assertEqual(actual['metric'], 'Cosine')
            self.assertLessEqual(len(actual['hits']), 10)
            self.assertTrue(all(h['score'] > 0 for h in actual['hits']))

    def test_pmid_markers_and_summary_are_derived_not_invented(self):
        a0 = compare_five_method_retrieval(self.index, self.query)
        pooled = {h['pmid'] for r in a0['rankings'].values() for h in r['hits']}
        self.assertEqual(set(a0['labels']), pooled)
        self.assertEqual(len(set(a0['labels'].values())), len(pooled))
        self.assertEqual(a0['summary']['unique_top10_pmids'], len(pooled))
        top1 = [r['hits'][0]['pmid'] for r in a0['rankings'].values() if r['hits']]
        self.assertEqual(a0['summary']['all_five_top1_same'],
                         len(top1) == 5 and len(set(top1)) == 1)
        for pair in a0['summary']['pairwise_overlap']:
            left = a0['rankings'][pair['Method A']]['hits']
            right = a0['rankings'][pair['Method B']]['hits']
            for n in (3, 10):
                actual = len({h['pmid'] for h in left[:n]} & {h['pmid'] for h in right[:n]})
                self.assertEqual(pair[f'Top {n} shared'], actual)
        self.assertEqual(len(a0['summary']['pairwise_overlap']), 10)
        self.assertEqual(len(a0['pmid_to_index']), len(pooled))
        for pmid, pos in a0['pmid_to_index'].items():
            self.assertEqual(self.index.docs[pos]['pmid'], pmid)

    def test_a0_selected_hit_powers_all_three_charts(self):
        a0 = compare_five_method_retrieval(self.index, self.query)
        chosen_pmid = a0['rankings']['Raw TF']['hits'][0]['pmid']
        pos = a0['pmid_to_index'][chosen_pmid]
        summaries, weights = compare_query_document_terms(
            self.index, self.query, pos, ['semaglutide', 'weight'])
        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(weights), 10)
        for term in ('semaglutide', 'weight'):
            part = [row for row in weights if row['Term'] == term]
            self.assertEqual(len(part), 5)
            self.assertEqual(len(build_single_term_five_weight_chart(part).data), 1)
            self.assertEqual(len(build_query_document_comparison_chart(part).data), 2)
        fig = build_two_term_five_weight_chart(weights)
        self.assertTrue(all(trace.orientation == 'h' for trace in fig.data))
        self.assertEqual([d.name for d in fig.data], ['semaglutide', 'weight'])

    def test_explicit_invalid_selection_no_silent_fallback(self):
        pool = compare_five_method_retrieval(self.index, 'semaglutide')['pmid_to_index']
        self.assertEqual(selected_pmid_after_query_change('12345', 'Top 10 selection', pool, True),
                         (None, True))
        pmid = next(iter(pool))
        self.assertEqual(selected_pmid_after_query_change(pmid, 'Top 10 selection', pool, True),
                         (pmid, False))
        self.assertEqual(selected_pmid_after_query_change('12345', 'PMID', pool, True),
                         ('12345', False))
        self.assertEqual(selected_pmid_after_query_change('12345', 'Top 10 selection', pool, False),
                         ('12345', False))

    def test_oov_query_is_empty_not_placeholder_ranks(self):
        a0 = compare_five_method_retrieval(self.index, 'totallyunknownqword')
        for result in a0['rankings'].values():
            self.assertFalse(result['hits'])
        self.assertEqual(a0['summary']['unique_top10_pmids'], 0)
        self.assertFalse(a0['summary']['all_five_top1_same'])
        self.assertEqual(a0['labels'], {})

    def test_formal_top10_limit_not_changed(self):
        with self.assertRaises(ValueError):
            compare_five_method_retrieval(self.index, self.query, limit=3)

    def test_01_core_remains_single_search_and_6_pages_preserved(self):
        source = (SRC / 'm09_app.py').read_text(encoding='utf-8')
        part_01 = source.split('if page == "01 Retrieval":')[1].split('elif page == "02 Weighting Comparison":')[0]
        self.assertIn('index.search(query,method,metric,limit=10)', part_01)
        part_02 = source.split('elif page == "02 Weighting Comparison":')[1].split('elif page == "03 Controlled Experiments":')[0]
        self.assertIn('compare_five_method_retrieval(index, comparison_query)', part_02)
        self.assertNotIn('Retrieval method for document selection', part_02)
        self.assertIn('wcmp_selected_pmid', part_02)
        self.assertIn('Full Top 10', part_02)
        self.assertNotIn('compare_embeddings(', part_02)
        tree = ast.parse(source)
        self.assertTrue(any(isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'MAIN_PAGES' for t in n.targets)
            for n in tree.body))


if __name__ == '__main__':
    unittest.main()
