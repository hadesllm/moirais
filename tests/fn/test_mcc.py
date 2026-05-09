"""Tests for moirais.fn.mcc -- Matthews Correlation Coefficient."""

import numpy as np
import pytest
from moirais.fn.mcc import matthews_corrcoef


class TestMCC:
    def test_perfect(self):
        y = np.array([0, 0, 1, 1])
        assert matthews_corrcoef(y, y) == pytest.approx(1.0, abs=1e-10)

    def test_inverse(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert matthews_corrcoef(y_true, y_pred) == pytest.approx(-1.0, abs=1e-10)

    def test_random_near_zero(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=1000)
        y_pred = rng.integers(0, 2, size=1000)
        mcc_val = matthews_corrcoef(y_true, y_pred)
        assert -0.15 < mcc_val < 0.15
