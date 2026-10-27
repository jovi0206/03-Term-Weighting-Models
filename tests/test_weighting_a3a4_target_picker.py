"""A3/A4 UI contraction and click-to-fill recommendation regression."""
import ast
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
SOURCE = (SRC / "m09_app.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def callable_from_source(name, state):
    function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {"st": SimpleNamespace(session_state=state)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<extracted callback>', 'exec'), env)
    return env[name]


class TestTermRecommendationClick(unittest.TestCase):
    def test_default_target_b_is_one_click_and_marks_manual(self):
        state = {"wcmp_term1": "glp-1", "wcmp_term2": "diabetes", "wcmp_b_user_edited": False}
        callback = callable_from_source('_wcmp_apply_recommended_term', state)
        callback('neprilysin')
        self.assertEqual(state['wcmp_term1'], 'glp-1')
        self.assertEqual(state['wcmp_term2'], 'neprilysin')
        self.assertTrue(state['wcmp_b_user_edited'])

    def test_explicit_target_a_changes_only_a(self):
        state = {"wcmp_recommend_target": "Term A", "wcmp_term1": "glp-1",
                 "wcmp_term2": "neprilysin", "wcmp_b_user_edited": True}
        callback = callable_from_source('_wcmp_apply_recommended_term', state)
        callback('measurement')
        self.assertEqual(state['wcmp_term1'], 'measurement')
        self.assertEqual(state['wcmp_term2'], 'neprilysin')
        self.assertTrue(state['wcmp_b_user_edited'])

    def test_explicit_target_b_does_not_change_a(self):
        state = {"wcmp_recommend_target": "Term B", "wcmp_term1": "glp-1",
                 "wcmp_term2": "", "wcmp_b_user_edited": False}
        callback = callable_from_source('_wcmp_apply_recommended_term', state)
        callback('diabetes')
        self.assertEqual(state['wcmp_term1'], 'glp-1')
        self.assertEqual(state['wcmp_term2'], 'diabetes')
        self.assertTrue(state['wcmp_b_user_edited'])

    def test_reset_b_returns_to_auto_suggestion(self):
        state = {"wcmp_term2": "some-other-term", "wcmp_b_user_edited": True,
                 "wcmp_first_recommended_b": "neprilysin"}
        callback = callable_from_source('_wcmp_reset_to_recommended_b', state)
        callback()
        self.assertEqual(state['wcmp_term2'], 'neprilysin')
        self.assertFalse(state['wcmp_b_user_edited'])


class TestA3A4Layout(unittest.TestCase):
    def setUp(self):
        self.page02 = SOURCE.split('elif page == "02 Weighting Comparison":', 1)[1].split(
            'elif page == "03 Controlled Experiments":', 1)[0]

    def test_default_a_one_time_migration_and_session_retention(self):
        self.assertIn('if "wcmp_defaults_a3a4_v3" not in st.session_state:', self.page02)
        self.assertIn('st.session_state["wcmp_term1"] = "glp-1"', self.page02)
        self.assertIn('st.session_state["wcmp_recommend_target"] = "Term B"', self.page02)
        self.assertIn('key="wcmp_term1"', self.page02)
        self.assertIn('key="wcmp_term2"', self.page02)

    def test_recommendation_buttons_select_current_target(self):
        self.assertIn('"Fill recommended term into"', self.page02)
        self.assertIn('st.button(row["Term"]', self.page02)
        self.assertIn('on_click=_wcmp_apply_recommended_term', self.page02)
        self.assertIn('args=(row["Term"],)', self.page02)
        self.assertIn('"Document TF", "Corpus DF", "IDF (log10)"', self.page02)
        self.assertIn('"Resume automatic Term B"', self.page02)

    def test_remove_a3_and_renumber_remaining_charts(self):
        self.assertNotIn('build_single_term_five_weight_chart(', self.page02)
        self.assertNotIn('A5 ·', self.page02)
        self.assertNotIn('#### A3 · Query weight vs Document weight', self.page02)
        self.assertNotIn('#### A4 · Two terms × five methods', self.page02)
        self.assertIn('#### A3 · Two terms × five methods · same document', self.page02)
        self.assertEqual(self.page02.count('st.plotly_chart(build_query_document_comparison_chart('), 0)
        self.assertEqual(self.page02.count('st.plotly_chart(build_two_term_five_weight_chart('), 1)

    def test_core_method_and_other_pages_untouched(self):
        # The 04 Evaluation display is intentionally updated. Verify that
        # the 05 Vector Projection and 06 Word2Vec pages remain identical.
        suffix = SOURCE.split('elif page == "05 Vector Projection":', 1)[1]
        self.assertEqual(hashlib.sha256(suffix.encode('utf-8')).hexdigest(),
                         '203f7eca1b82efe088d30682045927f5f8119aa38a3406b5f8e9405f17685a67')
        self.assertIn('cached_shared_umap_query(', self.page02)
        self.assertIn('compare_five_method_retrieval(index, comparison_query)', self.page02)
        self.assertIn('compare_query_document_terms(', self.page02)

if __name__ == '__main__':
    unittest.main()
