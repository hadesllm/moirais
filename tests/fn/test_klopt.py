"""Tests for morie.fn.klopt — KL divergence optimization."""

import numpy as np
import pytest

from morie.fn.klopt import klopt


class TestKlopt:
    def test_identity_projection(self):
        target = np.array([0.2, 0.3, 0.5])
        support = np.array([0.0, 1.0, 2.0])
        stats = [lambda x: x, lambda x: x**2]
        result = klopt(target, support, stats)
        assert result["kl_divergence"] >= 0

    def test_pmf_sums_to_one(self):
        target = np.array([0.4, 0.3, 0.2, 0.1])
        support = np.arange(4, dtype=float)
        result = klopt(target, support, [lambda x: x])
        assert np.sum(result["projected_pmf"]) == pytest.approx(1.0, abs=1e-6)

    def test_kl_nonnegative(self):
        target = np.array([0.5, 0.3, 0.2])
        support = np.arange(3, dtype=float)
        result = klopt(target, support, [lambda x: x])
        assert result["kl_divergence"] >= -1e-10

    def test_output_keys(self):
        result = klopt(
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0]),
            [lambda x: x],
        )
        assert "projected_pmf" in result
        assert "kl_divergence_bits" in result
        assert "lambdas" in result
