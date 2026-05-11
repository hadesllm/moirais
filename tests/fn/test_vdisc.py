"""Tests for vdisc -- discriminant validity (HTMT)."""
import numpy as np
from morie.fn.vdisc import discriminant_validity
from morie.fn._containers import DescriptiveResult


class TestDiscriminantValidity:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        R = np.corrcoef(X, rowvar=False)
        subs = {"F1": [0, 1, 2, 3, 4], "F2": [5, 6, 7, 8, 9]}
        result = discriminant_validity(R, subs)
        assert isinstance(result, DescriptiveResult)

    def test_htmt_col(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 6))
        R = np.corrcoef(X, rowvar=False)
        subs = {"A": [0, 1, 2], "B": [3, 4, 5]}
        result = discriminant_validity(R, subs)
        assert "HTMT" in result.value.columns
