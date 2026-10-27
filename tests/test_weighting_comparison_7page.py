"""Chart contracts and TF/DF/IDF comparisons; no formal corpus required."""
import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from e4_03_term_weighting import WEIGHTINGS, tf_weight
from e4_05_weighting_comparison import compare_query_document_terms, suggest_document_term_pair
from project3_charts import (
    build_single_term_five_weight_chart,
    build_query_document_comparison_chart,
    build_two_term_five_weight_chart, build_grouped_evaluation_chart,
)


def tiny_index():
    # 3-document synthetic corpus, not the user's actual 3,000-document corpus.
    # distinct TF and DF for visible A1/A2/A3 contrasts
    df = {"semaglutide": 2, "glp-1": 1, "weight": 2}
    vocab = {v: i for i, v in enumerate(sorted(df))}
    return SimpleNamespace(
        condition="C", docs=[{"pmid": "1"}, {"pmid": "2"}, {"pmid": "3"}],
        counters=[Counter({"semaglutide": 4, "glp-1": 1, "weight": 2}),
                  Counter({"semaglutide": 1, "weight": 1}), Counter()],
        vocabulary=vocab, df=df, cf={"semaglutide": 5, "glp-1": 1, "weight": 3},
        idfs=[math.log10(3 / df[t]) for t in sorted(df)],
    )


class TestLearningCharts(unittest.TestCase):
    def setUp(self):
        self.index = tiny_index()
        self.summaries, self.all_rows = compare_query_document_terms(
            self.index, "semaglutide semaglutide weight loss", 0, ["semaglutide", "glp-1"])
        self.rows_a = [r for r in self.all_rows if r["Term"] == "semaglutide"]

    def test_a1_single_horizontal_shared_axis_and_labels(self):
        chart = build_single_term_five_weight_chart(self.rows_a)
        self.assertEqual(len(chart.data), 1)
        self.assertEqual(chart.data[0].orientation, "h")
        self.assertEqual(list(chart.data[0].y), list(WEIGHTINGS))
        self.assertEqual(list(chart.data[0].x), [r["Document Weight"] for r in self.rows_a])
        self.assertTrue(all(x for x in chart.data[0].text))
        self.assertEqual(chart.layout.yaxis.categoryarray, tuple(reversed(WEIGHTINGS)))
        self.assertGreater(chart.layout.xaxis.range[1], max(chart.data[0].x))
        self.assertEqual(chart.data[0].x[0], 4.)  # raw TF

    def test_a2_grouped_horizontal_query_vs_doc(self):
        chart = build_query_document_comparison_chart(self.rows_a)
        self.assertEqual(chart.layout.barmode, "group")
        self.assertEqual(len(chart.data), 2)
        self.assertEqual(list(chart.data[0].x), [r["Query Weight"] for r in self.rows_a])
        self.assertEqual(list(chart.data[1].x), [r["Document Weight"] for r in self.rows_a])
        self.assertTrue(all(trace.orientation == "h" for trace in chart.data))
        self.assertEqual(chart.data[0].x[2], chart.data[1].x[2])  # both present -> IDF-only equal
        self.assertNotEqual(chart.data[0].x[0], chart.data[1].x[0])  # different TF
        self.assertNotEqual(chart.data[0].marker.color, chart.data[1].marker.color)
        self.assertLess(chart.layout.legend.y, 0)  # legend below chart

    def test_a3_one_grouped_chart_two_terms(self):
        chart = build_two_term_five_weight_chart(self.all_rows)
        self.assertEqual(len(chart.data), 2)
        self.assertEqual(chart.layout.barmode, "group")
        self.assertEqual(list(chart.data[0].y), list(WEIGHTINGS))
        self.assertEqual(list(chart.data[1].y), list(WEIGHTINGS))
        self.assertEqual(chart.data[0].x[0], 4.)
        self.assertEqual(chart.data[1].x[0], 1.)
        self.assertNotEqual(chart.data[0].x[2], chart.data[1].x[2])
        self.assertTrue(all(t.orientation == "h" for t in chart.data))
        self.assertTrue(all(t.text for t in chart.data))
        self.assertLess(chart.layout.legend.y, 0)  # legend below chart; never in title

    def test_absent_terms_zero_and_oov_remains_undefined(self):
        summaries, rows = compare_query_document_terms(self.index, "weight", 0, ["semaglutide", "liraglutide"])
        self.assertEqual(summaries[0]["Query TF"], 0)
        self.assertEqual(summaries[1]["Document TF"], 0)
        self.assertIsNone(summaries[1]["IDF (log10)"])
        self.assertTrue(all(r["Query Weight"] == 0 for r in rows))
        self.assertTrue(all(r["Document Weight"] == 0 for r in rows if r["Term"] == "liraglutide"))
        self.assertEqual(list(build_single_term_five_weight_chart(rows[:5]).data[0].x),
                         [r["Document Weight"] for r in rows[:5]])

    def test_recommendation_only_present_terms(self):
        a, b = suggest_document_term_pair(self.index, 0)
        self.assertNotEqual(a, b)
        self.assertGreater(self.index.counters[0][a], 0)
        self.assertGreater(self.index.counters[0][b], 0)
        self.assertEqual({a, b}, {"semaglutide", "glp-1"})

    def test_reject_malformed_or_negative_chart_values(self):
        with self.assertRaises(ValueError):
            build_single_term_five_weight_chart(self.rows_a[:4])
        bad = [dict(row) for row in self.rows_a]
        bad[0]["Document Weight"] = -1
        with self.assertRaises(ValueError):
            build_single_term_five_weight_chart(bad)
        with self.assertRaises(ValueError):
            build_two_term_five_weight_chart(self.all_rows[:-1])

    def test_existing_grouped_evaluation_chart_not_changed(self):
        means = [{"Weighting": m, "Precision@10": .5, "Recall@10": .2,
                  "MRR@10": .75, "nDCG@10": .9} for m in WEIGHTINGS]
        fig = build_grouped_evaluation_chart(means)
        self.assertEqual(fig.layout.barmode, "group")
        self.assertEqual(len(fig.data), 5)
        self.assertEqual(list(fig.data[0].x),
                         ["Precision@10", "Recall@10", "MRR@10", "nDCG@10"])

    def test_page_navigation_and_word2vec_separation(self):
        text = (SRC / "m09_app.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        pages_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "MAIN_PAGES" for t in n.targets))
        expected = ("01 Retrieval", "02 Weighting Comparison",
                    "03 Controlled Experiments", "04 Evaluation",
                    "05 Vector Projection", "06 Word2Vec Comparison")
        self.assertEqual(ast.literal_eval(pages_node.value), expected)
        lexical_segment = text.split('elif page == "02 Weighting Comparison":')[1].split('elif page == "03 Controlled Experiments":')[0]
        self.assertNotIn("model_inventory()", lexical_segment)
        self.assertNotIn("compare_embeddings(", lexical_segment)
        word2vec_segment = text.split('elif page == "06 Word2Vec Comparison":')[1]
        self.assertIn("model_inventory()", word2vec_segment)
        self.assertIn("compare_embeddings(", word2vec_segment)


if __name__ == "__main__":
    unittest.main()
