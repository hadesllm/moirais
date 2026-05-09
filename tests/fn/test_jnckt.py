"""Tests for jnckt (Jonckheere-Terpstra test)."""

import numpy as np
import pytest
from moirais.fn.jnckt import jnckt


class TestJnckt:
    """Jonckheere-Terpstra test for ordered alternatives."""

    def test_jnckt_ordered_samples(self):
        """Ordered samples should reject."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])
        result = jnckt(x, y, z)
        assert result["p_value"] < 0.05

    def test_jnckt_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = jnckt(x, y)
        required_keys = {"statistic", "z_stat", "p_value", "k", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_jnckt_single_sample_error(self):
        """Fewer than 2 samples should raise error."""
        with pytest.raises(ValueError):
            jnckt(np.array([1, 2, 3]))
