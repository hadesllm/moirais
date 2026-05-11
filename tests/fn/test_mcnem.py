"""Tests for mcnem (McNemar's test)."""

import numpy as np
import pytest
from morie.fn.mcnem import mcnem


class TestMcnem:
    """McNemar's test for paired binary data."""

    def test_mcnem_no_discordant(self):
        """No discordant pairs should not reject."""
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 0, 1, 1])
        result = mcnem(x, y)
        assert result["p_value"] == 1.0

    def test_mcnem_asymmetric_discordant(self):
        """Asymmetric discordant should reject."""
        x = np.array([0, 0, 0, 1, 1, 1])
        y = np.array([0, 1, 1, 1, 1, 1])
        result = mcnem(x, y)
        assert "p_value" in result

    def test_mcnem_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])
        result = mcnem(x, y)
        required_keys = {"statistic", "p_value", "n_01", "n_10", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_mcnem_length_mismatch_error(self):
        """x and y must have equal length."""
        with pytest.raises(ValueError):
            mcnem(np.array([0, 1, 0]), np.array([1, 0]))
