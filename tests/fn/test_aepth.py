"""Tests for morie.fn.aepth — AEP verification."""

import numpy as np
import pytest

from morie.fn.aepth import aepth


class TestAepth:
    def test_mean_converges_to_entropy(self):
        pmf = np.array([0.5, 0.5])
        result = aepth(pmf, 100, n_samples=5000)
        assert result["mean_neg_log_rate"] == pytest.approx(1.0, abs=0.05)

    def test_typical_fraction_high_for_large_n(self):
        pmf = np.array([0.3, 0.7])
        result = aepth(pmf, 500, n_samples=2000, epsilon=0.1)
        assert result["typical_fraction"] > 0.8

    def test_entropy_correct(self):
        pmf = np.array([0.25, 0.25, 0.25, 0.25])
        result = aepth(pmf, 10)
        assert result["entropy"] == pytest.approx(2.0, abs=1e-10)

    def test_deterministic(self):
        pmf = np.array([0.6, 0.4])
        r1 = aepth(pmf, 50, seed=99)
        r2 = aepth(pmf, 50, seed=99)
        assert r1["typical_fraction"] == r2["typical_fraction"]

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            aepth(np.array([0.5, 0.5]), 0)

    def test_output_keys(self):
        result = aepth(np.array([0.5, 0.5]), 10)
        assert "entropy" in result
        assert "mean_neg_log_rate" in result
        assert "typical_fraction" in result
