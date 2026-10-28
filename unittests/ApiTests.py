#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | API Tests
===================================

API tests. Uses Flask's built-in test client, so no server needs to be
running. Every request uses "mode": "test" so only test models and test
logs are used.
"""

import unittest

from unittests import *  # noqa: F401,F403
from app import app


class ApiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        # make sure test models exist before the predict tests run
        cls.train_response = cls.client.post("/train", json={"mode": "test"})

    def _predict(self, query, mode="test"):
        return self.client.post("/predict", json={"query": query, "mode": mode})

    def test_01_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")
        self.assertEqual(r.get_json()["project"], "AAVAIL Revenue Forecast")

    def test_02_train(self):
        self.assertEqual(self.train_response.status_code, 200)
        body = self.train_response.get_json()
        self.assertTrue(body["trained"])
        self.assertTrue(body["test"])
        self.assertIn("all", body["best_models"])

    def test_03_predict_all_countries(self):
        r = self._predict({"country": "all", "year": "2018", "month": "01", "day": "05"})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(len(body["y_pred"]), 1)
        self.assertGreaterEqual(body["y_pred"][0], 0)

    def test_04_predict_single_country(self):
        r = self._predict({"country": "united_kingdom", "year": "2019", "month": "05", "day": "01"})
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(r.get_json()["y_pred"][0], 0)

    def test_05_predict_country_name_with_space(self):
        r = self._predict({"country": "United Kingdom", "year": "2019", "month": "05", "day": "01"})
        self.assertEqual(r.status_code, 200)

    def test_06_predict_empty_body(self):
        r = self.client.post("/predict")
        self.assertEqual(r.status_code, 400)

    def test_07_predict_missing_keys(self):
        r = self._predict({"country": "all"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("missing", r.get_json()["error"])

    def test_08_predict_unknown_country(self):
        r = self._predict({"country": "atlantis", "year": "2018", "month": "01", "day": "05"})
        self.assertEqual(r.status_code, 400)

    def test_09_predict_date_out_of_range(self):
        r = self._predict({"country": "all", "year": "2030", "month": "01", "day": "01"})
        self.assertEqual(r.status_code, 400)

    def test_10_logs(self):
        r = self.client.get("/logs/train-test.log")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"unique_id", r.data)
        r = self.client.get("/logs/predict-test.log")
        self.assertEqual(r.status_code, 200)

    def test_11_logs_not_found(self):
        r = self.client.get("/logs/does-not-exist.log")
        self.assertEqual(r.status_code, 404)

    def test_12_monitor(self):
        r = self.client.get("/monitor?mode=test")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        for key in ("outlier_pct", "wasserstein_X", "production_rmse", "retrain_recommended"):
            self.assertIn(key, body)


if __name__ == "__main__":
    unittest.main()
