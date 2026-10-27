"""Evaluation UI presentation-only regression (synthetic means, no formal corpus)."""
import ast
from pathlib import Path
import hashlib
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
sys.path.insert(0,str(SRC))
from e4_03_term_weighting import WEIGHTINGS
from project3_charts import EVALUATION_METRICS,build_grouped_evaluation_chart

BASELINE_EVAL_SOURCE_SHA256='9a1a7cf7269b33caa2f48062c3aef28f0b914fbc74c78a5fe4beb7808db06530' # replaced below if different

class TestEvaluationUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.means=[{'Weighting':method,
                    'Precision@10':0.86 + i*.01,
                    'Recall@10':0.0327 + i*.009,
                    'MRR@10':0.9 + i*.015,
                    'nDCG@10':1.0 if i==4 else 0.83+i*.03}
                   for i,method in enumerate(WEIGHTINGS)]
        cls.fig=build_grouped_evaluation_chart(cls.means)
        cls.source=(SRC/'m09_app.py').read_text(encoding='utf-8')
        cls.page04=cls.source.split('elif page == "04 Evaluation":',1)[1].split('elif page == "05 Vector Projection":',1)[0]

    def test_five_grouped_series_in_original_weight_order(self):
        fig=self.fig
        self.assertEqual(fig.layout.barmode,'group')
        self.assertEqual([b.name for b in fig.data],list(WEIGHTINGS))
        self.assertEqual(list(fig.data[0].x),list(EVALUATION_METRICS))
        for i,bar in enumerate(fig.data):
            self.assertEqual(list(bar.y),[self.means[i][k] for k in EVALUATION_METRICS])
            self.assertEqual(len(bar.y),4)

    def test_title_legend_large_text_and_axis_bounds(self):
        fig=self.fig
        self.assertEqual(fig.layout.title.text,'Average Evaluation Scores across 10 Queries')
        self.assertEqual(fig.layout.yaxis.title.text,'Mean Score')
        self.assertEqual(tuple(fig.layout.yaxis.range),(0,1))
        self.assertGreaterEqual(fig.layout.legend.font.size,17)
        self.assertGreater(fig.layout.legend.y,1)
        self.assertGreaterEqual(fig.layout.xaxis.tickfont.size,18)
        self.assertGreaterEqual(fig.layout.yaxis.tickfont.size,16)

    def test_all_twenty_exact_four_decimal_labels_including_tiny_recall(self):
        fig=self.fig
        self.assertEqual(sum(len(tr.text) for tr in fig.data),20)
        for bar,mean in zip(fig.data,self.means):
            self.assertEqual(list(bar.text),[f'{mean[m]:.4f}' for m in EVALUATION_METRICS])
            self.assertEqual(bar.textposition,'outside')
            self.assertFalse(bar.cliponaxis)
            self.assertGreaterEqual(bar.textfont.size,14)
        self.assertEqual(self.fig.data[0].text[1],'0.0327')
        self.assertEqual(self.fig.data[-1].text[-1],'1.0000')

    def test_ui_uses_existing_evaluation_data_and_actual_qrels_count(self):
        p=self.page04
        self.assertIn('rows, means, qrels = cached_eval(',p)
        self.assertIn('build_grouped_evaluation_chart(means)',p)
        self.assertIn('"Why is Recall@10 relatively low?"',p)
        self.assertIn("len(qrels[example_rule['id']])",p)
        self.assertIn('min(10, relevant_count) / relevant_count',p)
        self.assertIn('semaglutide weight loss',p)
        self.assertIn('"How to interpret the four IR metrics"',p)
        self.assertIn('"Export mean CSV"',p)
        self.assertIn('"Export proxy qrels JSON"',p)
        self.assertIn('**Per-query evaluation**',p)
        self.assertIn('automated abstract-topic regex PROXIES',p)
        self.assertNotIn('Recall@50',p)
        self.assertNotIn('Recall@100',p)

    def test_formal_evaluation_code_unchanged(self):
        # Direct SHA of the existing evaluator. No scoring or qrels modifications.
        content=(SRC/'e7_01_retrieval_evaluation.py').read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(),BASELINE_EVAL_SOURCE_SHA256)

    def test_new_ui_is_only_in_evaluation_page(self):
        a=self.source.split('elif page == "04 Evaluation":',1)[0]
        z=self.source.split('elif page == "05 Vector Projection":',1)[1]
        self.assertNotIn('Why is Recall@10 relatively low?',a+z)
        self.assertNotIn('How to interpret the four IR metrics',a+z)

    def test_other_pages_exactly_unchanged(self):
        # Shared header colors and titles are allowed to change. The actual
        # 01–03 and 05–06 page implementations must remain byte-for-byte the
        # same as the known-good Evaluation release.
        pages_01_to_03 = self.source.split('if page == "01 Retrieval":', 1)[1].split('elif page == "04 Evaluation":', 1)[0]
        # Only 02/03 section-heading RQ suffixes were intentionally removed.
        # Restore just those headings FOR COMPARISON to the approved SHA-256;
        # any other change within 01-03 must still fail this test.
        allowed_titles = (
            ('section("Weighting Comparison",', 'section("Weighting Comparison · RQ1–RQ3",'),
            ('section("Controlled Experiments",', 'section("Controlled Experiments · RQ4",'),
        )
        for new_title, prior_title in allowed_titles:
            self.assertEqual(pages_01_to_03.count(new_title), 1)
            self.assertNotIn(prior_title, pages_01_to_03)
            pages_01_to_03 = pages_01_to_03.replace(new_title, prior_title, 1)
        # The Evaluation heading must also be the requested presentation-only wording.
        self.assertIn('section("Retrieval Evaluation",', self.source)
        suffix = self.source.split('elif page == "05 Vector Projection":', 1)[1]
        self.assertEqual(hashlib.sha256(pages_01_to_03.encode()).hexdigest(),
                         'fb3167b705a3d07e3ba290443682094a2d0f21afb7df8875adda662d31c4abd9')
        self.assertEqual(hashlib.sha256(suffix.encode()).hexdigest(),
                         '203f7eca1b82efe088d30682045927f5f8119aa38a3406b5f8e9405f17685a67')

if __name__=='__main__': unittest.main()
