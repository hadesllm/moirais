"""Tests for morie.fn.cusum — CUSUM change detection."""

import numpy as np
import pytest

from morie.fn.cusum import cusum


class TestCusum:
    """Tests for cusum()."""

    def test_detects_mean_shift(self):
        """Detects a clear mean shift."""
        series = np.concatenate([
            np.zeros(50),
            np.full(50, 5.0),
        ])
        result = cusum(series, target_mean=0.0, threshold=10.0)
        assert len(result["change_points"]) > 0
        # Change point should be near index 50
        assert any(45 <= cp <= 60 for cp in result["change_points"])

    def test_no_change_in_constant(self):
        """No change detected in a constant series."""
        series = np.full(100, 3.0)
        result = cusum(series, threshold=5.0)
        assert len(result["change_points"]) == 0

    def test_arrays_correct_length(self):
        """CUSUM arrays match series length."""
        series = np.random.default_rng(42).standard_normal(80)
        result = cusum(series)
        assert len(result["cusum_pos"]) == 80
        assert len(result["cusum_neg"]) == 80
