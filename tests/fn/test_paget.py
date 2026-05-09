"""Tests for paget (Page's test for ordered alternatives)."""

import numpy as np
import pytest
from moirais.fn.paget import paget


class TestPaget:
    """Page's test for ordered alternatives."""

    def test_paget_ordered_treatments(self):
        """Ordered treatments should reject (need enough blocks for power)."""
        data = np.array([[i + j for j in range(4)] for i in range(8)], dtype=float)
        result = paget(data)
        assert result["p_value"] < 0.05

    def test_paget_returns_dict(self):
        """Return type should be dict with required keys."""
        data = np.random.default_rng(42).standard_normal((4, 3))
        result = paget(data)
        required_keys = {"statistic", "z_stat", "p_value", "k", "b", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_paget_shape_error(self):
        """1D input should raise error."""
        with pytest.raises(ValueError):
            paget(np.array([1, 2, 3, 4]))

    def test_paget_small_input_error(self):
        """Need ≥2 blocks and ≥2 treatments."""
        with pytest.raises(ValueError):
            paget(np.array([[1, 2]]))
