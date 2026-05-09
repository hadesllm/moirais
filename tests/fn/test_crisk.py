"""Tests for moirais.fn.crisk — competing risks CIF."""

import numpy as np
import pytest

from moirais.fn.crisk import crisk


class TestCrisk:
    """Tests for crisk()."""

    def test_two_event_types(self):
        """CIF returned for two competing event types."""
        rng = np.random.default_rng(42)
        n = 100
        time = rng.exponential(5, size=n)
        event = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        result = crisk(time, event)
        assert 1 in result["cif"]
        assert 2 in result["cif"]
        assert len(result["cif"][1]) == len(result["times"])

    def test_cif_bounded(self):
        """CIF values are non-negative and sum <= 1 at each time."""
        rng = np.random.default_rng(7)
        n = 200
        time = rng.exponential(3, size=n)
        event = rng.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3])
        result = crisk(time, event)
        cif_sum = result["cif"][1] + result["cif"][2]
        assert np.all(cif_sum >= -1e-10)
        assert np.all(cif_sum <= 1.0 + 1e-10)

    def test_raises_all_censored(self):
        """All-censored data raises ValueError."""
        with pytest.raises(ValueError, match="No events"):
            crisk(np.array([1, 2, 3]), np.array([0, 0, 0]))
