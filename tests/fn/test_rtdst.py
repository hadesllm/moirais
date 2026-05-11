"""Tests for morie.fn.rtdst — rate-distortion function."""

import numpy as np
import pytest

from morie.fn.rtdst import rtdst


class TestRtdst:
    def test_binary_source_zero_distortion(self):
        pmf = np.array([0.5, 0.5])
        D = np.array([[0, 1], [1, 0]])
        result = rtdst(pmf, D, 0.0)
        assert result["rate"] >= 0.0

    def test_high_distortion_zero_rate(self):
        pmf = np.array([0.5, 0.5])
        D = np.array([[0, 1], [1, 0]])
        result = rtdst(pmf, D, 10.0)
        assert result["rate"] == pytest.approx(0.0, abs=1e-6)

    def test_rate_decreases_with_distortion(self):
        pmf = np.array([0.3, 0.7])
        D = np.array([[0, 1], [1, 0]])
        r1 = rtdst(pmf, D, 0.05)
        r2 = rtdst(pmf, D, 0.2)
        assert r1["rate"] >= r2["rate"] - 1e-6

    def test_invalid_pmf(self):
        with pytest.raises(ValueError):
            rtdst(np.array([0.3, 0.3]), np.array([[0, 1], [1, 0]]), 0.1)

    def test_negative_distortion(self):
        with pytest.raises(ValueError):
            rtdst(np.array([0.5, 0.5]), np.array([[0, 1], [1, 0]]), -1.0)

    def test_output_keys(self):
        result = rtdst(np.array([0.5, 0.5]), np.array([[0, 1], [1, 0]]), 0.3)
        assert "rate" in result
        assert "distortion" in result
        assert "optimal_mapping" in result
