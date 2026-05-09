"""Tests for moirais.fn.kmise — MISE-optimal bandwidth."""

import numpy as np
import pytest

from moirais.fn.kmise import kmise


class TestKmise:
    def test_formula_normal(self):
        data = np.arange(1, 101, dtype=float)
        res = kmise(data)
        sigma = np.std(data, ddof=1)
        expected = (4.0 * sigma ** 5 / (3.0 * 100)) ** 0.2
        assert res["bw_opt"] == pytest.approx(expected, rel=1e-6)

    def test_positive_mise(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        res = kmise(data)
        assert res["mise_approx"] > 0

    def test_bw_decreases_with_n(self):
        rng = np.random.default_rng(42)
        small = rng.normal(0, 1, 50)
        large = rng.normal(0, 1, 500)
        r1 = kmise(small)
        r2 = kmise(large)
        assert r2["bw_opt"] < r1["bw_opt"]

    def test_raises_unsupported_kernel(self):
        with pytest.raises(ValueError):
            kmise(np.array([1.0, 2.0, 3.0]), kernel="epanechnikov")

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kmise(np.array([1.0]))
