#!/usr/bin/env python
"""
AAVAIL Revenue Forecast | Test Runner
=====================================

Run ALL unit tests (API, model, logging) with one command:

    python run-tests.py            # all tests
    python run-tests.py -v         # verbose
    python run-tests.py api        # only one suite: api | model | logger

Tests only use test models (models/test/) and test logs (logs/test/).
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SUITES = {"api": "ApiTests.py", "model": "ModelTests.py", "logger": "LoggerTests.py"}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbosity = 2 if "-v" in sys.argv else 1
    pattern = SUITES.get(args[0].lower(), "*Tests.py") if args else "*Tests.py"

    suite = unittest.TestLoader().discover(os.path.join(HERE, "unittests"),
                                           pattern=pattern, top_level_dir=HERE)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
