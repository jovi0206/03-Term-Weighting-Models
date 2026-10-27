"""Oct 26 A1–A5 offline regression, synthetic docs; NOT a 3k runtime claim."""
import ast
import gzip
import html
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

from e4_03_term_weighting import WEIGHTINGS, tf_weight
from e4_05_weighting_comparison import compare_query_document_terms
from e4_06_weighting_retrieval_comparison import compare_five_method_retrieval
from e4_07_document_term_recommendation import (recommend_document_terms,
                                                  diagnose_document_term,
                                                  term_b_on_document_change)
from m08_controller import SearchIndex
from project3_charts import (build_single_term_five_weight_chart,
    build_query_document_comparison_chart, build_two_term_five_weight_chart)


class TestA1A5Update(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        p = Path(cls.temp.name) / 'synthetic.jsonl.gz'
        records = [
            {'pmid': '301', 'title': 'Diabetes semaglutide cardiovascular treatment',
             'abstract_clean': 'glp-1 glp-1 glp-1 semaglutide semaglutide '
             'diabetes diabetes cardiovascular obesity insulin receptor'},
            {'pmid': '302', 'title': 'Neprilysin title mention only',
             'abstract_clean': 'glp-1 liraglutide diabetes pancreatic pancreatitis kidney obesity'},
            {'pmid': '303', 'title': 'Renal adverse reactions',
             'abstract_clean': 'glp-1 metabolic cardiovascular gastric pancreatic diabetes'},
            {'pmid': '304', 'title': 'GLP-1 receptor agonist',
             'abstract_clean': 'glp-1 glucagon receptor agonist semaglutide cardiac insulin'},
            {'pmid': '305', 'title': 'Gastric adverse event',
             'abstract_clean': 'glp-1 insulin gastric treatment diabetes semaglutide'},
            {'pmid': '306', 'title': 'Obesity management',
             'abstract_clean': 'glp-1 obesity pancreatic inflammation metabolic diabetes'},
        ]
        with gzip.open(p, 'wt', encoding='utf-8') as out:
            out.write(json.dumps({'record_type': 'project2_corpus_header',
                                  'corpus_sha256': 'synthetic'}) + '\n')
            for record in records:
                out.write(json.dumps(record) + '\n')
        cls.index = SearchIndex.build(corpus_path=p, condition='C')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_top_five_recommendations_are_real_abstract_terms(self):
        results = recommend_document_terms(self.index, 0, term_a='glp-1')
        self.assertGreaterEqual(len(results), 1)
        self.assertLessEqual(len(results), 5)
        for record in results:
            t = record['Term']
            self.assertNotEqual(t, 'glp-1')
            self.assertEqual(record['Document TF'], self.index.counters[0][t])
            self.assertGreater(record['Document TF'], 0)
            self.assertEqual(record['Corpus DF'], self.index.df[t])
            self.assertEqual(record['Corpus CF'], self.index.cf[t])
            self.assertAlmostEqual(record['IDF (log10)'],
                                   self.index.idfs[self.index.vocabulary[t]])
            self.assertEqual(record['Higher IDF than Term A'],
                             record['IDF (log10)'] >
                             self.index.idfs[self.index.vocabulary['glp-1']])
        self.assertEqual(results,
                         recommend_document_terms(self.index, 0, term_a='glp-1'))

    def test_recommendation_moves_with_selected_pmid(self):
        a = recommend_document_terms(self.index, 0)
        b = recommend_document_terms(self.index, 1)
        self.assertNotEqual([x['Term'] for x in a], [x['Term'] for x in b])
        self.assertTrue(all(self.index.counters[1][x['Term']] > 0 for x in b))

    def test_recommendation_excludes_manually_changed_term_a(self):
        self.assertNotIn('semaglutide', [row['Term'] for row in
                         recommend_document_terms(self.index, 0, 'semaglutide')])

    def test_recommendation_uses_unchanged_weights(self):
        for row in recommend_document_terms(self.index, 0):
            terms, weight_rows = compare_query_document_terms(
                self.index, 'glp-1', 0, ['glp-1', row['Term']])
            second = terms[1]
            self.assertEqual(second['Document TF'], row['Document TF'])
            self.assertAlmostEqual(second['IDF (log10)'], row['IDF (log10)'])
            self.assertEqual(len(weight_rows), 2 * len(WEIGHTINGS))
            log = [rec for rec in weight_rows if rec['Term'] == row['Term']
                   and rec['Weighting'] == 'Log TF-IDF'][0]
            self.assertAlmostEqual(log['Document Weight'],
                tf_weight(row['Document TF'], 'Log TF-IDF', row['IDF (log10)']))

    def test_tf0_reason_title_only(self):
        d = diagnose_document_term(self.index, 1, 'neprilysin')
        self.assertEqual(d['pmid'], '302')
        self.assertEqual(d['document_tf'], 0)
        # term is absent from vocabulary, so OOV is correctly prioritized.
        self.assertIn('OOV', d['reason'])
        # Corpus known but article-specific absent: not index/data corruption.
        d = diagnose_document_term(self.index, 1, 'semaglutide')
        self.assertEqual(d['corpus_df'], 3)
        self.assertEqual(d['document_tf'], 0)
        self.assertIn('Main Abstract', d['reason'])

    def test_selected_pmid_same_index_as_retrieval(self):
        result = compare_five_method_retrieval(self.index, 'glp-1')
        for pmid, row in result['pmid_to_index'].items():
            self.assertEqual(str(self.index.docs[row]['pmid']), pmid)

    def test_top3_color_classification_matches_summary(self):
        result = compare_five_method_retrieval(self.index, 'glp-1')
        repeated = set(result['summary']['top3_repeated_pmids'])
        for method in WEIGHTINGS:
            for hit in result['rankings'][method]['hits'][:3]:
                pmid = str(hit['pmid'])
                appearance = sum(pmid in {str(h['pmid']) for h in
                    result['rankings'][m]['hits'][:3]} for m in WEIGHTINGS)
                self.assertEqual(pmid in repeated, appearance >= 2)

    def test_manual_term_b_never_overwritten_on_document_change(self):
        a = ('sha', 'C', '301', 'glp-1')
        b = ('sha', 'C', '302', 'glp-1')
        self.assertEqual(term_b_on_document_change(None, a, '', False, 'insulin'), 'insulin')
        self.assertEqual(term_b_on_document_change(a, a, 'semaglutide', False, 'insulin'), 'semaglutide')
        self.assertEqual(term_b_on_document_change(a, b, 'semaglutide', True, 'pancreatic'), 'semaglutide')
        self.assertEqual(term_b_on_document_change(a, b, 'insulin', False, 'pancreatic'), 'pancreatic')

    def test_a1_card_no_rainbow_and_correct_classes(self):
        source = (SRC / 'm09_app.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_a1_card')
        env = {'html': html}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<a1card>', 'exec'), env)
        hit = {'pmid':'123', 'rank':1, 'title':'Some abstract', 'score':0.2}
        self.assertIn('wcmp-a1-card repeated', env['_a1_card'](hit, 'D01', False, True))
        self.assertIn('wcmp-a1-card unique', env['_a1_card'](hit, 'D02', False, False))
        self.assertNotIn('hsl(', env['_a1_card'](hit, 'D03', False, True))

    def test_three_horizontal_charts_projector_readable(self):
        _, rows = compare_query_document_terms(self.index, 'glp-1', 0,
                                                ['glp-1', 'semaglutide'])
        a = rows[:5]
        figs = (build_single_term_five_weight_chart(a),
                build_query_document_comparison_chart(a),
                build_two_term_five_weight_chart(rows))
        for fig in figs:
            self.assertTrue(all(tr.orientation == 'h' for tr in fig.data))
            self.assertTrue(all(tr.textfont.size >= 17 for tr in fig.data))
            self.assertGreaterEqual(fig.layout.yaxis.tickfont.size, 17)
            self.assertIsNone(fig.layout.yaxis.title.text)
            self.assertGreater(fig.layout.margin.r, 100)
            self.assertGreater(fig.layout.xaxis.range[1], max(
                float(x) for tr in fig.data for x in tr.x))
        self.assertGreaterEqual(figs[1].layout.legend.font.size, 17)
        self.assertLess(figs[1].layout.legend.y, 0)
        self.assertEqual(figs[2].data[0].name, 'glp-1')

    def test_only_02_default_query_changed_and_methods_intact(self):
        src = (SRC / 'm09_app.py').read_text(encoding='utf-8')
        part = src.split('elif page == "02 Weighting Comparison":')[1].split(
            'elif page == "03 Controlled Experiments":')[0]
        self.assertIn('st.session_state["wcmp_active_query_v2"] = "glp-1"', part)
        self.assertIn('wcmp_b_user_edited', part)
        self.assertIn('Recommended Terms from Selected Document', part)
        self.assertIn('Unique PMIDs across Top 10', part)
        self.assertIn('repeated=pmid in top3_repeated', part)
        self.assertIn('cached_shared_umap(', part)
        self.assertIn('cached_shared_umap_query(', part)
        self.assertIn('"wcmp_selected_pmid"', src)
        self.assertEqual(len(WEIGHTINGS), 5)


if __name__ == '__main__':
    unittest.main()
