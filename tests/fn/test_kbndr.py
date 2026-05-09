"""Tests for moirais.fn.kbndr — boundary-corrected KDE."""

import numpy as np
import pytest

from moirais.fn.kbndr import kbndr


class TestKbndr:
    def test_nonneg_at_boundary(self):
        rng = np.random.default_rng(42)
        data = np.abs(rng.normal(0, 1, 300))
        res = kbndr(data, lower=0.0)
        assert np.all(res["density"] >= -1e-12)

    def test_reflection_method(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(2.0, 200)
        res = kbndr(data, lower=0.0, method="reflection")
        assert res["method"] == "reflection"
        assert res["density"][0] >= 0

    def test_renormalization_method(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, 200)
        res = kbndr(data, lower=0.0, upper=1.0, method="renormalization")
        assert res["method"] == "renormalization"

    def test_raises_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown"):
            kbndr(np.array([1.0, 2.0, 3.0]), method="bogus")

    def test_raises_small(self):
        with pytest.raises(ValueError):
            kbndr(np.array([1.0]))
