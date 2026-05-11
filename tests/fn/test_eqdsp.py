"""Tests for morie.fn.eqdsp — disparity decomposition."""

import pytest
import numpy as np
from morie.fn.eqdsp import disparity_decompose
from morie.fn._containers import DescriptiveResult


class TestDisparityDecompose:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X_a = rng.normal(5, 1, (100, 2))
        X_b = rng.normal(3, 1, (100, 2))
        ya = X_a @ [2, 1] + rng.normal(0, 1, 100)
        yb = X_b @ [2, 1] + rng.normal(0, 1, 100)
        r = disparity_decompose(ya, X_a, yb, X_b)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["total_gap"] > 0

    def test_too_few(self):
        with pytest.raises(ValueError):
            disparity_decompose([1, 2], np.array([[1], [2]]), [1], np.array([[1]]))
