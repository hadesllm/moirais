"""Tests for areff (asymptotic relative efficiency)."""

import numpy as np
import pytest
from morie.fn.areff import areff


class TestAreff:
    """Asymptotic relative efficiency."""

    def test_areff_basic(self):
        """Basic ARE computation."""
        result = areff(100, 120, 50)
        assert result["are"] == 1.2

    def test_areff_returns_dict(self):
        """Return type should be dict with required keys."""
        result = areff(100, 100, 50)
        required_keys = {"are", "interpretation", "n_ratio"}
        assert set(result.keys()) == required_keys

    def test_areff_invalid_stats_error(self):
        """Negative statistics should raise error."""
        with pytest.raises(ValueError):
            areff(-100, 100, 50)
