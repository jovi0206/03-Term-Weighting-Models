"""02 final Top10 and A3 consolidation: UI-only, synthetic retrieval hits."""
from pathlib import Path
import ast
import html
import hashlib
import sys
import unittest
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
sys.path.insert(0,str(SRC))
from e4_03_term_weighting import WEIGHTINGS
from e4_08_top10_presentation import build_five_top10_view

APP=(SRC/'m09_app.py').read_text(encoding='utf8')
PAGE=APP.split('elif page == "02 Weighting Comparison":',1)[1].split('elif page == "03 Controlled Experiments":',1)[0]
TREE=ast.parse(APP)
RENDER_NODE=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='_wcmp_top10_table_html')
RENDER_ENV={'html':html}
exec(compile(ast.Module(body=[RENDER_NODE],type_ignores=[]),'<render>','exec'),RENDER_ENV)
render=RENDER_ENV['_wcmp_top10_table_html']


def fake_results(prefix='q'):
    # Five methods, 10 positive actual mock hits each. Some repeat, some do not.
    results={}
    for m,method in enumerate(WEIGHTINGS):
        results[method]={"hits":[{"rank":i+1,
          "pmid":str(9000+i) if i<4 else str(10000 + m*100 + i),
          "title":f"Abstract title for {prefix} with a <tag> & details {m}/{i}",
          "score": 0.18 + 0.003*m + 0.001*(10-i)}
          for i in range(10)]}
    pmids=sorted({str(h['pmid']) for r in results.values() for h in r['hits']})
    markers={p:f'D{j+1:02d}' for j,p in enumerate(pmids)}
    return results,markers

class TestTop10FinalUI(unittest.TestCase):
    def setUp(self):
        self.results,self.markers=fake_results()
        self.view=build_five_top10_view(self.results,self.markers)

    def test_fifty_rows_five_methods_one_source(self):
        self.assertEqual(tuple(self.view['tables']),tuple(WEIGHTINGS))
        self.assertEqual(len(self.view['csv_rows']),50)
        for method in WEIGHTINGS:
            self.assertEqual([r['Rank'] for r in self.view['tables'][method]],list(range(1,11)))
            self.assertEqual([r['Cosine Score'] for r in self.view['tables'][method]],
                             [h['score'] for h in self.results[method]['hits']])

    def test_repeated_uses_distinct_method_membership_not_rank(self):
        self.assertEqual(self.view['repeated_pmids'],frozenset(str(9000+i) for i in range(4)))
        for rows in self.view['tables'].values():
            for row in rows:
                self.assertEqual(row['Repeated'],row['PMID'] in self.view['repeated_pmids'])

    def test_d_markers_consistent_with_a1_pool(self):
        for method in WEIGHTINGS:
            for row in self.view['tables'][method]:
                self.assertEqual(row['D ID'],self.markers[row['PMID']])

    def test_full_title_and_score_exposed_with_safe_html(self):
        rows=self.view['tables'][WEIGHTINGS[0]]
        markup=render(WEIGHTINGS[0], rows)
        for label in ('Rank','D ID','PMID','Title','Cosine Score'):
            self.assertIn(f'>{label}</th>',markup)
        self.assertIn('0.190000',markup)
        self.assertIn('title="Abstract title for q with a &lt;tag&gt; &amp; details',markup)
        self.assertNotIn('<tag>',markup)
        self.assertEqual(markup.count('<tr class='),10)
        self.assertIn('class="repeated"',markup)
        self.assertIn('class="unique"',markup)

    def test_query_change_does_not_cache_or_reuse_old_results(self):
        other,markers=fake_results(prefix='q2')
        other[WEIGHTINGS[1]]['hits'][0]['score']=0.654321
        next_view=build_five_top10_view(other,markers)
        self.assertNotEqual(next_view['tables'][WEIGHTINGS[1]][0]['Cosine Score'],
                            self.view['tables'][WEIGHTINGS[1]][0]['Cosine Score'])
        self.assertIn('q2',next_view['tables'][WEIGHTINGS[1]][0]['Title'])

    def test_empty_positive_hit_list_not_faked(self):
        empty={m:{'hits':[]} for m in WEIGHTINGS}
        view=build_five_top10_view(empty,{})
        self.assertEqual(len(view['csv_rows']),0)
        self.assertEqual(len(view['repeated_pmids']),0)
        self.assertIn('No positive-score matches',render('Raw TF',view['tables']['Raw TF']))

    def test_reject_mismatched_ids_scores_and_ranks(self):
        r,m=fake_results()
        r[WEIGHTINGS[0]]['hits'][0]['pmid']='no-marker'
        with self.assertRaisesRegex(ValueError,'Missing A1'):
            build_five_top10_view(r,m)
        r,m=fake_results()
        r[WEIGHTINGS[0]]['hits'][0]['rank']=7
        with self.assertRaisesRegex(ValueError,'ranks'):
            build_five_top10_view(r,m)
        r,m=fake_results()
        r[WEIGHTINGS[0]]['hits'][0]['score']=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            build_five_top10_view(r,m)

    def test_top10_layout_three_then_two_and_exports_preserved(self):
        self.assertIn('first_row = st.columns(3',PAGE)
        self.assertIn('zip(first_row, WEIGHTINGS[:3])',PAGE)
        self.assertIn('zip(second_row[1:3], WEIGHTINGS[3:])',PAGE)
        self.assertIn('csv_bytes(top10_view["csv_rows"])',PAGE)
        self.assertIn('"Export weighting comparison CSV"',PAGE)
        self.assertIn('wcmp_full_top10_pmid',PAGE)
        self.assertIn('on_click=_select_weighting_document',PAGE)
        self.assertIn('top10_view = build_five_top10_view(ranked, markers)',PAGE)

    def test_a3_two_terms_only_with_stats_retained(self):
        self.assertIn('### A3 · Two Terms × Five Methods',PAGE)
        self.assertIn('#### A3 · Two terms × five methods · same document',PAGE)
        self.assertNotIn('#### A3 · Query weight vs Document weight',PAGE)
        self.assertNotIn('#### A4 · Two terms',PAGE)
        self.assertNotIn('wcmp_focus',PAGE)
        self.assertEqual(PAGE.count('st.plotly_chart(build_two_term_five_weight_chart('),1)
        self.assertIn('compare_query_document_terms(',PAGE)
        self.assertIn('#### Exact source values · TF / CF / DF / IDF',PAGE)
        self.assertIn('Recommended Terms from Selected Document',PAGE)

    def test_other_pages_and_navigation_unchanged(self):
        old=Path(__file__).resolve().parent.parent/'src/m09_app.py'
        self.assertIn('elif page == "06 Word2Vec Comparison":',APP)
        self.assertIn('elif page == "05 Vector Projection":',APP)
        self.assertIn('elif page == "04 Evaluation":',APP)
        self.assertIn('elif page == "03 Controlled Experiments":',APP)

if __name__=='__main__': unittest.main()
