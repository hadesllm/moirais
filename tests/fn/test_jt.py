"""Tests for moirais.fn.jt -- Jonckheere-Terpstra test."""

import numpy as np
import pytest
from moirais.fn.jt import jonckheere_terpstra_test
from moirais.fn._containers import TestResult


class TestJT:
    def test_ordered_groups(self):
        """Groups with clear ordering => significant."""
        r = jonckheere_terpstra_test([1, 2, 3], [4, 5, 6], [7, 8, 9])
        assert isinstance(r, TestResult)
        assert r.p_value < 0.05

    def test_no_trend(self):
        """Overlapping groups => non-significant."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 30)
        b = rng.normal(0, 1, 30)
        c = rng.normal(0, 1, 30)
        r = jonckheere_terpstra_test(a, b, c)
        assert r.p_value > 0.05

    def test_raises_single_group(self):
        with pytest.raises(ValueError):
            jonckheere_terpstra_test([1, 2, 3])
