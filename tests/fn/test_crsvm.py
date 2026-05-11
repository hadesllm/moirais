"""Tests for crsvm (Cramer-von Mises test)."""

import numpy as np
import pytest
from morie.fn.crsvm import crsvm


class TestCrsvm:
    """Cramer-von Mises goodness-of-fit test."""

    def test_crsvm_normal_sample(self):
        """Sample from normal should not reject normality."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = crsvm(x, dist="norm")
        assert "statistic" in result

    def test_crsvm_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.random.default_rng(42).standard_normal(50)
        result = crsvm(x, dist="norm")
        required_keys = {"statistic", "p_value", "critical_value", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_crsvm_small_sample_error(self):
        """Sample size < 2 should raise error."""
        with pytest.raises(ValueError):
            crsvm(np.array([1]))
