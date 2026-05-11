"""Tests for rnkci (rank-based confidence intervals)."""

import numpy as np
import pytest
from morie.fn.rnkci import rnkci


class TestRnkci:
    """Rank-based confidence intervals for location."""

    def test_rnkci_basic(self):
        """Basic CI computation."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = rnkci(x)
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_rnkci_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.random.default_rng(42).standard_normal(20)
        result = rnkci(x, ci_level=0.95)
        required_keys = {"point_estimate", "ci_lower", "ci_upper", "ci_level", "n"}
        assert set(result.keys()) == required_keys

    def test_rnkci_small_sample_error(self):
        """Sample size < 3 should raise error."""
        with pytest.raises(ValueError):
            rnkci(np.array([1, 2]))
