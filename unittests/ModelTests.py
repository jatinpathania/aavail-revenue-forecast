#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Model Tests
=====================================

Model tests. All models are trained with test=True, so they are written to
models/test/ with the 'test' prefix and never overwrite production models.
"""

import os
import unittest

from unittests import *  # noqa: F401,F403  (sets sys.path)
from model import (model_train, model_load, model_predict, compare_models,
                   TEST_MODEL_DIR, MODEL_DIR, MODEL_VERSION, TEST_PREFIX, PROD_PREFIX)
from cslib import fetch_ts, engineer_features, DATA_DIR

VERSION = str(MODEL_VERSION).replace(".", "_")


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        prod_files = sorted(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []
        cls.prod_mtimes = {f: os.path.getmtime(os.path.join(MODEL_DIR, f))
                           for f in prod_files if f.endswith(".joblib")}
        cls.comparison = model_train(test=True)

    def test_01_train(self):
        """test models are saved in the isolated test folder"""
        for tag in ("all", "united_kingdom"):
            path = os.path.join(TEST_MODEL_DIR, f"{TEST_PREFIX}-{tag}-{VERSION}.joblib")
            self.assertTrue(os.path.exists(path), path)

    def test_02_production_models_untouched(self):
        """training in test mode must not modify production models"""
        for f, mtime in self.prod_mtimes.items():
            self.assertEqual(mtime, os.path.getmtime(os.path.join(MODEL_DIR, f)))
            self.assertTrue(f.startswith(PROD_PREFIX))

    def test_03_multiple_models_compared(self):
        """every candidate (including the baseline) is evaluated"""
        models = set(self.comparison["model"])
        self.assertTrue({"baseline", "linear", "random_forest",
                         "gradient_boosting"}.issubset(models))
        self.assertTrue((self.comparison["rmse"] >= 0).all())

    def test_04_load(self):
        all_data, all_models = model_load(test=True)
        self.assertIn("all", all_models)
        for saved in all_models.values():
            self.assertTrue(hasattr(saved["model"], "predict"))
        self.assertIn("X", all_data["all"])

    def test_05_predict_all(self):
        result = model_predict("all", "2018", "01", "05", test=True)
        self.assertEqual(len(result["y_pred"]), 1)
        self.assertGreaterEqual(float(result["y_pred"][0]), 0)

    def test_06_predict_country(self):
        result = model_predict("United Kingdom", "2019", "06", "30", test=True)
        self.assertGreaterEqual(float(result["y_pred"][0]), 0)

    def test_07_predict_bad_country(self):
        with self.assertRaises(ValueError):
            model_predict("atlantis", "2018", "01", "05", test=True)

    def test_08_predict_bad_date(self):
        with self.assertRaises(ValueError):
            model_predict("all", "2030", "01", "01", test=True)

    def test_09_features(self):
        ts = fetch_ts(DATA_DIR)["all"]
        X, y, dates = engineer_features(ts, training=True)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(ts) - 30)
        self.assertFalse(X.isnull().any().any())

    def test_10_compare_models_returns_sorted(self):
        ts = fetch_ts(DATA_DIR)["all"]
        X, y, _ = engineer_features(ts)
        results, fitted = compare_models(X, y)
        self.assertEqual(list(results["rmse"]), sorted(results["rmse"]))
        self.assertEqual(set(fitted), set(results["model"]))


if __name__ == "__main__":
    unittest.main()
