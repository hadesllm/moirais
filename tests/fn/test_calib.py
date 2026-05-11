"""Tests for morie.fn.calib -- Calibration curve."""

import numpy as np
import pytest
from morie.fn.calib import calibration_curve


class TestCalibrationCurve:
    def test_perfect_calibration(self):
        """Perfectly calibrated model: observed freq = predicted prob."""
        rng = np.random.default_rng(42)
        n = 2000
        y_prob = rng.random(n)
        y_true = (rng.random(n) < y_prob).astype(float)
        result = calibration_curve(y_true, y_prob, n_bins=5)
        assert "bin_means" in result
        assert "bin_freqs" in result
        assert "brier_score" in result
        # Bin means and freqs should be roughly equal
        for m, f in zip(result["bin_means"], result["bin_freqs"]):
            assert abs(m - f) < 0.15

    def test_brier_score_perfect(self):
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_prob = np.array([0, 0, 1, 1], dtype=float)
        result = calibration_curve(y_true, y_prob)
        assert result["brier_score"] == pytest.approx(0.0, abs=1e-10)

    def test_bad_bins_raises(self):
        with pytest.raises(ValueError):
            calibration_curve([0, 1], [0.5, 0.5], n_bins=0)
