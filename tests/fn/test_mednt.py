"""Tests for mednt (median test - two sample)."""

import numpy as np
import pytest
from moirais.fn.mednt import mednt


class TestMednt:
    """Median test for two independent samples."""

    def test_mednt_identical_samples(self):
        """Identical samples should not reject."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = mednt(x, y)
        assert result["p_value"] > 0.05

    def test_mednt_different_medians(self):
        """Samples with different medians should reject."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 11, 12, 13, 14])
        result = mednt(x, y)
        assert result["p_value"] < 0.05

    def test_mednt_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([6, 7, 8, 9, 10])
        result = mednt(x, y)
        required_keys = {
            "statistic",
            "p_value",
            "n1_above",
            "n1_below",
            "n2_above",
            "n2_below",
            "interpretation",
        }
        assert set(result.keys()) == required_keys

    def test_mednt_contingency_sums(self):
        """Sums should not exceed total sample sizes."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])
        result = mednt(x, y)
        assert result["n1_above"] + result["n1_below"] <= len(x)
        assert result["n2_above"] + result["n2_below"] <= len(y)

    def test_mednt_empty_sample_error(self):
        """Empty sample should raise error."""
        with pytest.raises(ValueError):
            mednt(np.array([1, 2, 3]), np.array([]))
