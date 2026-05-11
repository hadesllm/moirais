"""Statistics is the grammar of science. — Karl Pearson"""

import numpy as np
from morie.fn.rdpil import red_pill_test, rdpil
from morie.fn._containers import TestResult


class TestRdpil:
    def test_alias(self):
        assert rdpil is red_pill_test

    def test_reject_null(self):
        x = np.array([10, 11, 12, 10, 11, 12, 10, 11, 12, 10], dtype=float)
        result = red_pill_test(x, mu0=0.0)
        assert isinstance(result, TestResult)
        assert result.extra["decision"] == "red_pill"

    def test_fail_to_reject(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30)
        result = red_pill_test(x, mu0=0.0)
        assert result.extra["decision"] == "blue_pill"
