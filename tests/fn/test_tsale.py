"""Tests for morie.fn.tsale — Tsallis entropy."""

import numpy as np
import pytest

from morie.fn.tsale import tsale


class TestTsale:
    def test_q1_is_shannon_nats(self):
        pmf = np.array([0.3, 0.7])
        result = tsale(pmf, 1.0)
        expected = -0.3 * np.log(0.3) - 0.7 * np.log(0.7)
        assert result["entropy"] == pytest.approx(expected, abs=1e-10)

    def test_q2(self):
        pmf = np.array([0.5, 0.5])
        result = tsale(pmf, 2.0)
        expected = (1 - (0.25 + 0.25)) / (2 - 1)
        assert result["entropy"] == pytest.approx(expected, abs=1e-10)

    def test_uniform_maximal(self):
        pmf = np.array([0.25, 0.25, 0.25, 0.25])
        result = tsale(pmf, 2.0)
        assert result["entropy"] > 0

    def test_degenerate_zero(self):
        pmf = np.array([1.0, 0.0, 0.0])
        result = tsale(pmf, 2.0)
        assert result["entropy"] == pytest.approx(0.0, abs=1e-10)

    def test_invalid_q(self):
        with pytest.raises(ValueError):
            tsale(np.array([0.5, 0.5]), 0.0)
        with pytest.raises(ValueError):
            tsale(np.array([0.5, 0.5]), -1.0)

    def test_output_keys(self):
        result = tsale(np.array([0.5, 0.5]), 1.5)
        assert "entropy" in result
        assert "entropy_bits" in result
        assert "q" in result
        assert "shannon_entropy" in result
