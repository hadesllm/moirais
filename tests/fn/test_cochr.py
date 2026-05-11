"""Tests for cochr (Cochran's Q test)."""

import numpy as np
import pytest
from morie.fn.cochr import cochr


class TestCochr:
    """Cochran's Q test for k paired binary responses."""

    def test_cochr_identical_treatments(self):
        """Identical treatment effects should not reject."""
        data = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])
        result = cochr(data)
        assert result["p_value"] > 0.05

    def test_cochr_different_treatments(self):
        """Different treatments should reject."""
        data = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        result = cochr(data)
        assert result["p_value"] < 0.05

    def test_cochr_returns_dict(self):
        """Return type should be dict with required keys."""
        data = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1]])
        result = cochr(data)
        required_keys = {"statistic", "p_value", "k", "b", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_cochr_non_binary_error(self):
        """Non-binary data should raise error."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            cochr(data)
