#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Logging Tests
=======================================

Logging tests. All writes go to logs/test/ (test=True), never to the
production logs in logs/.
"""

import os
import csv
import shutil
import unittest

from unittests import *  # noqa: F401,F403
from logger import (update_train_log, update_predict_log, get_log_path,
                    LOG_DIR, TEST_LOG_DIR, TRAIN_HEADER, PREDICT_HEADER)


def _read(path):
    with open(path) as f:
        return list(csv.reader(f))


class LoggerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # start from a clean test log folder; production logs are not touched
        if os.path.isdir(TEST_LOG_DIR):
            shutil.rmtree(TEST_LOG_DIR)
        cls.prod_logs = {f: os.path.getsize(os.path.join(LOG_DIR, f))
                         for f in os.listdir(LOG_DIR) if f.endswith(".log")} \
            if os.path.isdir(LOG_DIR) else {}

    def test_01_test_paths_are_isolated(self):
        for kind in ("train", "predict"):
            test_path = get_log_path(kind, test=True)
            prod_path = get_log_path(kind, test=False)
            self.assertTrue(test_path.startswith(TEST_LOG_DIR))
            self.assertNotEqual(test_path, prod_path)

    def test_02_train_log_created(self):
        update_train_log("united_kingdom", ("2017-11-29", "2019-07-01"), "random_forest",
                         {"rmse": 0.5}, "000:00:01", 0.1, "unit test", test=True)
        self.assertTrue(os.path.exists(get_log_path("train", test=True)))

    def test_03_train_log_content(self):
        update_train_log("all", ("2017-11-29", "2019-07-01"), "linear",
                         {"rmse": 12.3}, "000:00:02", 0.1, "unit test", test=True)
        rows = _read(get_log_path("train", test=True))
        self.assertEqual(rows[0], TRAIN_HEADER)
        self.assertEqual(rows[-1][2], "all")
        self.assertIn("12.3", rows[-1][5])

    def test_04_predict_log_created(self):
        update_predict_log("all", [1234.5], None, "2018-01-05", "000:00:00", 0.1, test=True)
        self.assertTrue(os.path.exists(get_log_path("predict", test=True)))

    def test_05_predict_log_content(self):
        update_predict_log("germany", [99.9], None, "2018-02-01", "000:00:00", 0.1, test=True)
        rows = _read(get_log_path("predict", test=True))
        self.assertEqual(rows[0], PREDICT_HEADER)
        self.assertEqual(rows[-1][2], "germany")
        self.assertEqual(rows[-1][3], "2018-02-01")
        self.assertIn("99.9", rows[-1][4])

    def test_06_production_logs_untouched(self):
        for f, size in self.prod_logs.items():
            self.assertEqual(size, os.path.getsize(os.path.join(LOG_DIR, f)))

    def test_07_bad_log_kind(self):
        with self.assertRaises(ValueError):
            get_log_path("unknown", test=True)


if __name__ == "__main__":
    unittest.main()
