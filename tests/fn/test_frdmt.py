"""Tests for frdmt (Friedman two-way ANOVA)."""

import numpy as np
import pytest
from moirais.fn.frdmt import frdmt


class TestFrdmt:
    """Friedman two-way ANOVA by ranks."""

    def test_frdmt_identical_treatments(self):
        """Identical treatments should not reject."""
        data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        result = frdmt(data)
        assert result["p_value"] > 0.05

    def test_frdmt_different_treatments(self):
        """Very different treatments should reject."""
        data = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]])
        result = frdmt(data)
        assert result["p_value"] < 0.05

    def test_frdmt_returns_dict(self):
        """Return type should be dict with required keys."""
        data = np.random.default_rng(42).standard_normal((4, 3))
        result = frdmt(data)
        required_keys = {"statistic", "p_value", "k", "b", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_frdmt_shape_error(self):
        """1D input should raise error."""
        with pytest.raises(ValueError):
            frdmt(np.array([1, 2, 3, 4]))

    def test_frdmt_small_input_error(self):
        """Need ≥2 blocks and ≥2 treatments."""
        with pytest.raises(ValueError):
            frdmt(np.array([[1, 2]]))
