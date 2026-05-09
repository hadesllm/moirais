"""Tests for rtest -- test-retest reliability."""
import numpy as np
from moirais.fn.rtest import retest_reliability
from moirais.fn._containers import ESRes


class TestRetest:
    def test_perfect_agreement(self):
        scores = np.arange(20, dtype=float)
        result = retest_reliability(scores, scores)
        assert isinstance(result, ESRes)
        assert result.estimate > 0.99

    def test_ci_ordered(self):
        rng = np.random.default_rng(42)
        t1 = rng.standard_normal(50)
        t2 = t1 + rng.standard_normal(50) * 0.3
        result = retest_reliability(t1, t2)
        assert result.ci_lower <= result.estimate <= result.ci_upper
