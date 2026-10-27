"""UI-only six-page contraction regression tests (no live 3k document data)."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

APP = Path(__file__).resolve().parents[1] / 'src' / 'm09_app.py'
SOURCE = APP.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
EXPECTED = (
    '01 Retrieval',
    '02 Weighting Comparison',
    '03 Controlled Experiments',
    '04 Evaluation',
    '05 Vector Projection',
    '06 Word2Vec Comparison',
)


def navigation_constant():
    for stmt in TREE.body:
        if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'MAIN_PAGES' for t in stmt.targets):
            return ast.literal_eval(stmt.value)
    raise AssertionError('MAIN_PAGES missing')


class SixPageNavigation(TestCase):
    def test_exact_six_pages_and_no_removed_page_handler(self):
        self.assertEqual(navigation_constant(), EXPECTED)
        for title in EXPECTED:
            self.assertEqual(SOURCE.count('page == "' + title + '"'), 1)
        self.assertNotIn('page == "03 Term Weighting"', SOURCE)
        self.assertNotIn('Inspect term', SOURCE)
        self.assertNotIn('teacher_document_index', SOURCE)

    def test_no_orphan_or_incorrect_evaluation_page_reference(self):
        self.assertIn('see 04 Evaluation / Proxy Qrels', SOURCE)
        # Removed RQ4 help text is not required to carry a page link.
        self.assertIn('elif page == "04 Evaluation":', SOURCE)
        self.assertNotIn('Why Cosine and L2 Dot should agree', SOURCE)
        self.assertNotIn('on 05 Evaluation', SOURCE)
        self.assertNotIn('see 05 Evaluation', SOURCE)

    def test_analysis_pages_keep_unique_handlers(self):
        for header in ('Weighting Comparison',
                       'Controlled Experiments',
                       'Retrieval Evaluation',
                       '2-D Vector Space Projection',
                       'Word2Vec Comparison · optional learning demo'):
            self.assertIn(header, SOURCE)
        for old_title in ('Weighting Comparison · RQ1–RQ3',
                          'Controlled Experiments · RQ4',
                          'Retrieval Evaluation · RQ5'):
            self.assertNotIn('section("' + old_title + '"', SOURCE)
        self.assertIn('compare_five_method_retrieval(index, comparison_query)', SOURCE)
        self.assertNotIn('st.plotly_chart(build_query_document_comparison_chart(', SOURCE)
        self.assertIn('st.plotly_chart(build_two_term_five_weight_chart(', SOURCE)
        self.assertIn('cached_shared_umap_query(', SOURCE)

    def test_one_time_old_page_migration(self):
        top_nav = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'top_navigation')
        # Execute only the migration statements before selector selection.
        migrate = top_nav.body[:1]
        for old_page, intended in {
            '03 Term Weighting':'02 Weighting Comparison',
            '04 Controlled Experiments':'03 Controlled Experiments',
            '05 Evaluation':'04 Evaluation',
            '06 Vector Projection':'05 Vector Projection',
            '07 Word2Vec Comparison':'06 Word2Vec Comparison',
        }.items():
            state = {'project3_active_page': old_page}
            env={'st':SimpleNamespace(session_state=state)}
            exec(compile(ast.Module(body=migrate, type_ignores=[]), '<migration>', 'exec'), env)
            self.assertEqual(state['project3_active_page'], intended)
            self.assertTrue(state['project3_nav_six_migrated'])
            # No second migration after user chooses a new page with colliding old number.
            state['project3_active_page']='04 Evaluation'
            exec(compile(ast.Module(body=migrate, type_ignores=[]), '<migration>', 'exec'), env)
            self.assertEqual(state['project3_active_page'], '04 Evaluation')
