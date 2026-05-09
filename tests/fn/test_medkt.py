"""Tests for medkt (median test - k samples)."""

import numpy as np
import pytest
from moirais.fn.medkt import medkt


class TestMedkt:
    """Median test for k independent samples."""

    def test_medkt_identical_samples(self):
        """Identical samples should not reject."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        z = np.array([1, 2, 3, 4, 5])
        result = medkt(x, y, z)
        assert result["p_value"] > 0.05

    def test_medkt_different_samples(self):
        """Different samples should reject."""
        x = np.array([1, 2, 3])
        y = np.array([10, 11, 12])
        z = np.array([20, 21, 22])
        result = medkt(x, y, z)
        assert result["p_value"] < 0.05

    def test_medkt_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = medkt(x, y)
        required_keys = {"statistic", "p_value", "k", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_medkt_single_sample_error(self):
        """Fewer than 2 samples should raise error."""
        with pytest.raises(ValueError):
            medkt(np.array([1, 2, 3]))
