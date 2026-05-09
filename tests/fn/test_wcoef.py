"""Tests for wcoef (Kendall's coefficient of concordance)."""

import numpy as np
import pytest
from moirais.fn.wcoef import wcoef


class TestWcoef:
    """Kendall's coefficient of concordance W."""

    def test_wcoef_perfect_agreement(self):
        """Perfect agreement should have W = 1."""
        data = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        result = wcoef(data)
        assert result["concordance"] > 0.99

    def test_wcoef_returns_dict(self):
        """Return type should be dict with required keys."""
        data = np.random.default_rng(42).standard_normal((3, 4))
        result = wcoef(data)
        required_keys = {"concordance", "chi2_stat", "p_value", "k", "n", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_wcoef_small_input_error(self):
        """Need ≥2 judges and ≥2 objects."""
        with pytest.raises(ValueError):
            wcoef(np.array([[1, 2]]))
