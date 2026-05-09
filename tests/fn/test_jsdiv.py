"""Tests for moirais.fn.jsdiv — Jensen-Shannon divergence."""
import numpy as np
import pytest
from moirais.fn.jsdiv import js_divergence


class TestJSDivergence:
    def test_same_distribution(self):
        p = np.array([0.5, 0.5])
        res = js_divergence(p, p)
        assert res.estimate == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        r1 = js_divergence(p, q)
        r2 = js_divergence(q, p)
        assert r1.estimate == pytest.approx(r2.estimate, abs=1e-10)

    def test_bounded(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        res = js_divergence(p, q)
        assert res.estimate <= np.log(2) + 1e-10
