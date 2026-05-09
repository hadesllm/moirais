"""Tests for moirais.fn.shply — Shapley values."""
import numpy as np
import pytest
from moirais.fn.shply import shapley_value


class TestShapley:
    def test_additive_game(self):
        def v(S):
            return sum(i + 1 for i in S)
        res = shapley_value(v, 3)
        np.testing.assert_allclose(res.extra["values"], [1, 2, 3], atol=1e-10)

    def test_efficiency(self):
        def v(S):
            return len(S) ** 2
        res = shapley_value(v, 3)
        assert res.extra["values"].sum() == pytest.approx(v(frozenset({0, 1, 2})))
