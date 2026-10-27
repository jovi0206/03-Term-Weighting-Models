"""Project 3 header/background-only change assertions."""
from pathlib import Path
import ast
import unittest

APP = Path(__file__).resolve().parents[1] / 'src' / 'm09_app.py'

class TestProject3HeroStyle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.source)

    def test_header_text_matches_requested_style(self):
        self.assertIn('class="p3-kicker">Artificial Intelligence Information Retrieval</div>', self.source)
        self.assertIn('class="p3-title">WEIGHTING</div>', self.source)
        self.assertNotIn('NCKU · BIOMEDICAL INFORMATION RETRIEVAL</div>', self.source)
        self.assertNotIn('PROJECT 3 · TERM WEIGHTING & VECTOR SPACE RETRIEVAL</div>', self.source)
        self.assertNotIn('class="p3-sub"', self.source)

    def test_gradient_and_neutral_page_background(self):
        self.assertIn('linear-gradient(108deg,#1C2922 0%,#294332 55%,#4B7053 100%)', self.source)
        self.assertIn('[data-testid="stAppViewContainer"] {background-color:#F6F7F2 !important}', self.source)
        self.assertIn('.p3-title {font-size:44px', self.source)
        self.assertIn('@media (max-width:640px)', self.source)

    def test_workflow_labels_are_project3_not_project1(self):
        for name in ('3,000 Abstracts', 'Term Weighting', 'Vector Retrieval', 'Evaluation'):
            self.assertIn(f'class="p3-stage">{name}</span>', self.source)
        self.assertNotIn('Upload Files', self.source)
        self.assertNotIn('Locate Match', self.source)

    def test_navigation_and_page_implementation_are_preserved(self):
        main_pages = next(n for n in self.tree.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='MAIN_PAGES' for t in n.targets))
        self.assertEqual(ast.literal_eval(main_pages.value), (
          '01 Retrieval','02 Weighting Comparison','03 Controlled Experiments',
          '04 Evaluation','05 Vector Projection','06 Word2Vec Comparison'))
        self.assertIn('page = top_navigation()', self.source)
        self.assertIn('elif page == "04 Evaluation":', self.source)
        self.assertIn('elif page == "03 Controlled Experiments":', self.source)

if __name__ == '__main__':
    unittest.main()
