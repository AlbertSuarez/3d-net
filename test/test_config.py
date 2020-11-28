import os
import unittest

from src.config import DATASET_FOLDER, DATASET_CATEGORIES


class ConfigTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_existence(self):
        self.assertTrue(os.path.isdir(DATASET_FOLDER))

    def test_categories(self):
        self.assertEqual(len(DATASET_CATEGORIES.keys()), len(set(DATASET_CATEGORIES.keys())))
        self.assertTrue(all(k.count('__') == 1 for k in DATASET_CATEGORIES.keys()))
        self.assertTrue(all('category_id' in v for v in DATASET_CATEGORIES.values()))
        self.assertTrue(all('main_category' in v for v in DATASET_CATEGORIES.values()))
        self.assertTrue(all(k.startswith(f'{v.get("main_category")}__') for k, v in DATASET_CATEGORIES.items()))
        self.assertEqual(
            len([v.get('category_id') for v in DATASET_CATEGORIES.values()]),
            len(set([v.get('category_id') for v in DATASET_CATEGORIES.values()]))
        )
