"""Pure RQ4 view regression: no live 3,000-document corpus required."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e7_04_rq4_ranking_view import make_rq4_view, RQ4_METRICS


RAW = [str(x) for x in range(1, 11)]
COS = ["3", "9", "1", "11", "12", "13", "14", "15", "16", "17"]


def rows(pmids, delta=0.0):
    return [dict(rank=i, pmid=p, score=1/(i+1)+delta, title=f"Study about {p}",
                 abstract=f"Main Abstract {p}", pmcid="PMC"+p,
                 length=40+i, l2_norm=2.7, doc_index=i) for i,p in enumerate(pmids)]


class FakeIndex:
    corpus_sha256="fixed-corpus"
    def __init__(self, raw=RAW, cosine=COS, l2=COS, delta=1e-16):
        self.by_metric={"Raw Dot Product":rows(raw),
                        "Cosine":rows(cosine),
                        "L2 Dot Product":rows(l2,delta)}
        self.calls=[]
    def search(self, query, method, metric, k):
        self.calls.append((query,method,metric,k))
        return {"hits":self.by_metric[metric][:k], "oov":[]}


class TestThreeMetricView(unittest.TestCase):
    def test_exact_three_scoring_modes_and_fixed_controls(self):
        idx=FakeIndex()
        make_rq4_view(idx, "glp-1", "Raw TF-IDF", 10)
        self.assertEqual(idx.calls, [("glp-1","Raw TF-IDF",m,10) for m,_ in RQ4_METRICS])

    def test_top3_colors_repeat_across_any_two_of_three_methods(self):
        view=make_rq4_view(FakeIndex(),"glp-1")
        # 1/3 are shared Raw+Cosine; 9 is shared Cosine+L2 only.
        self.assertEqual(view["top3_repeated_pmids"], {"1", "3", "9"})
        self.assertNotIn("2", view["top3_repeated_pmids"])  # Raw-only = green
        self.assertEqual(view["raw_cos_top10_overlap"], 3)
        self.assertEqual(view["top10_repeated_pmids"], {"1", "3", "9"})
        # Color does not change independent metric scope or Top-10 list.

    def test_top3_color_cosine_l2_repeat_and_raw_unique(self):
        view=make_rq4_view(FakeIndex(
            raw=["21", "22", "23"],
            cosine=["11", "12", "13"],
            l2=["11", "12", "13"]), "glp-1")
        self.assertEqual(view["top3_repeated_pmids"], {"11", "12", "13"})
        self.assertEqual(view["raw_cos_top10_overlap"], 0)
        self.assertEqual(view["top10_repeated_pmids"], set())
        self.assertEqual(view["markers"]["11"], "D04")
        self.assertTrue(view["cos_l2_same_order"])

    def test_same_pmid_same_d_marker_across_methods(self):
        view=make_rq4_view(FakeIndex(),"glp-1")
        self.assertEqual(view["markers"]["3"],"D03")
        self.assertEqual(view["markers"]["9"],"D09")
        for metric, _ in RQ4_METRICS:
            for row in view["tables"][metric]:
                self.assertEqual(row["D ID"],view["markers"][row["PMID"]])

    def test_dynamic_cos_l2_ranking_and_score_error(self):
        view=make_rq4_view(FakeIndex(delta=6.94e-18),"glp-1")
        self.assertTrue(view["cos_l2_same_order"])
        self.assertGreaterEqual(view["max_cos_l2_error"],0.0)
        self.assertEqual(view["cos_l2_shared_pmids"],10)
        self.assertEqual(len(view["export_rows"]),30)

    def test_error_matches_pmids_not_row_positions(self):
        idx=FakeIndex(l2=list(reversed(COS)), delta=0.0)
        # Deliberately assign cosine PMID's own score even in shuffled L2 rankings.
        by_pmid={h['pmid']:h['score'] for h in idx.by_metric['Cosine']}
        for hit in idx.by_metric['L2 Dot Product']:
            hit['score']=by_pmid[hit['pmid']]
        view=make_rq4_view(idx,"glp-1")
        self.assertFalse(view["cos_l2_same_order"])
        self.assertEqual(view["max_cos_l2_error"],0.0)

    def test_switch_query_or_weighting_recomputes_all(self):
        idx=FakeIndex()
        make_rq4_view(idx,"glp-1", "Raw TF-IDF")
        make_rq4_view(idx,"diabetes", "Log TF-IDF")
        self.assertEqual(len(idx.calls),6)
        self.assertEqual(idx.calls[-1], ("diabetes","Log TF-IDF","L2 Dot Product",10))

    def test_empty_all_metrics_do_not_claim_equivalence(self):
        view=make_rq4_view(FakeIndex(raw=[],cosine=[],l2=[]),"unknown")
        self.assertTrue(view["no_positive_hits"])
        self.assertFalse(view["cos_l2_same_order"])
        self.assertIsNone(view["max_cos_l2_error"])
        self.assertEqual(view["export_rows"],[])

    def test_reject_duplicate_pmids_in_one_ranking(self):
        idx=FakeIndex(raw=['1','1','2'])
        with self.assertRaisesRegex(ValueError,"Repeated PMID"):
            make_rq4_view(idx,"glp-1")

    def test_reject_nonfinite_score(self):
        idx=FakeIndex()
        idx.by_metric["Cosine"][0]["score"]=float('nan')
        with self.assertRaisesRegex(ValueError,"Non-finite"):
            make_rq4_view(idx,"glp-1")

    def test_no_nonpositive_k(self):
        with self.assertRaises(ValueError):
            make_rq4_view(FakeIndex(),"glp-1",k=0)

    def test_03_main_ui_only_two_metrics_and_no_technical_expander(self):
        source=(ROOT / 'src/m09_app.py').read_text(encoding='utf-8')
        page=source.split('elif page == "03 Controlled Experiments":',1)[1].split(
            'elif page == "04 Evaluation":',1)[0]
        self.assertIn('Repeated PMID across methods',page)
        self.assertIn('Unique PMID',page)
        self.assertNotIn('In both Raw Dot and Cosine Top 3',page)
        self.assertIn('cols = st.columns(2, gap="medium")',page)
        self.assertIn('Raw Dot vs Cosine · Top 10 Overlap',page)
        self.assertIn('Cosine vs L2 Dot · Same Ordered Top 10',page)
        self.assertNotIn('Max |Cosine − L2 Dot|',page)
        self.assertNotIn('rq4["max_cos_l2_error"]',page)
        self.assertNotIn('Why Cosine and L2 Dot should agree',page)
        self.assertIn('Full Top 10 · Raw Dot / Cosine / L2 Dot',page)
        self.assertIn('Export RQ4 Top 10 CSV',page)
        # The full Top-10 table retains its explicit Raw/Cos highlight scope.
        self.assertIn('rq4["top10_repeated_pmids"]',page)

    def test_03_ui_scope_and_both_other_pages_preserved(self):
        source=(ROOT / 'src/m09_app.py').read_text(encoding='utf-8')
        sect=source.split('elif page == "03 Controlled Experiments":')[1].split('elif page == "04 Evaluation":')[0]
        self.assertIn('value="glp-1", key="rq4_query_v2"',sect)
        self.assertIn('WEIGHTINGS.index("Raw TF-IDF")',sect)
        self.assertIn('load_index("C", current_corpus_hash)',sect)
        self.assertIn('make_rq4_view(rq4_index, query.strip(), weighting, k=10)',sect)
        self.assertIn('rq4["top3_repeated_pmids"]',sect)
        self.assertIn('st.download_button("Export RQ4 Top 10 CSV"',sect)
        self.assertIn('Full Top 10 · Raw Dot / Cosine / L2 Dot',sect)
        self.assertIn('Inspect any Top 10 article',sect)
        self.assertNotIn('compare_rankings(',sect)
        self.assertNotIn('Method A',sect)
        self.assertNotIn('Method B',sect)
        self.assertIn('elif page == "02 Weighting Comparison":',source)
        self.assertIn('elif page == "04 Evaluation":',source)
        # 02 is intentionally revised for the five Top-10 tables and new A3.
        # Preserve the original approved 02-body hash; permit ONLY the
        # requested removal of the RQ suffix from its display heading.
        from hashlib import sha256
        page02=source.split('elif page == "02 Weighting Comparison":',1)[1].split(
            'elif page == "03 Controlled Experiments":',1)[0]
        current_heading = 'section("Weighting Comparison",'
        prior_heading = 'section("Weighting Comparison · RQ1–RQ3",'
        self.assertEqual(page02.count(current_heading), 1)
        self.assertNotIn(prior_heading, page02)
        page02_for_baseline = page02.replace(current_heading, prior_heading, 1)
        self.assertEqual(sha256(page02_for_baseline.encode()).hexdigest(),
                         'b88422686573856e8bf98677a5fb6eec9a68b4079bca5b1146fb36430bdcf3df')
        ast.parse(source)


if __name__=='__main__':
    unittest.main()
