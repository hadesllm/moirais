"""Tests for moirais.fn.wznrr — Wyner-Ziv bound."""

import numpy as np
import pytest

from moirais.fn.wznrr import wznrr


class TestWznrr:
    def test_side_info_helps(self):
        pxy = np.array([[0.4, 0.1], [0.1, 0.4]])
        D = np.array([[0, 1], [1, 0]])
        result = wznrr(pxy, D, 0.1)
        assert result["rate_wz"] <= result["rate_no_si"] + 1e-6

    def test_high_distortion_zero_rate(self):
        pxy = np.array([[0.5, 0.0], [0.0, 0.5]])
        D = np.array([[0, 1], [1, 0]])
        result = wznrr(pxy, D, 10.0)
        assert result["rate_wz"] == pytest.approx(0.0, abs=1e-10)

    def test_saving_nonnegative(self):
        pxy = np.array([[0.3, 0.2], [0.1, 0.4]])
        D = np.array([[0, 1], [1, 0]])
        result = wznrr(pxy, D, 0.2)
        assert result["rate_saving"] >= -1e-10

    def test_output_keys(self):
        pxy = np.array([[0.25, 0.25], [0.25, 0.25]])
        D = np.array([[0, 1], [1, 0]])
        result = wznrr(pxy, D, 0.3)
        assert "rate_wz" in result
        assert "H_X_given_Y" in result

    def test_invalid_negative_d(self):
        with pytest.raises(ValueError):
            wznrr(np.array([[0.5, 0.5]]), np.array([[0, 1]]), -1.0)
