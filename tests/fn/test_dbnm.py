"""Tests for moirais.fn.dbnm — binomial PMF."""

import numpy as np
import pytest

from moirais.fn.dbnm import dbinom


class TestDbinom:
    """Tests for dbinom()."""

    def test_fair_coin(self):
        """dbinom(5, 10, 0.5) ~ 0.2461 (C(10,5) * 0.5^10)."""
        assert dbinom(5, 10, 0.5) == pytest.approx(0.2461, abs=1e-3)

    def test_certain_event(self):
        """dbinom(0, 10, 0.0) = 1.0 (no successes when prob=0)."""
        assert dbinom(0, 10, 0.0) == pytest.approx(1.0, abs=1e-12)

    def test_type(self):
        """Returns float for scalar input."""
        result = dbinom(3, 10, 0.5)
        assert isinstance(result, (float, np.floating))

    def test_raises_bad_prob(self):
        """Should reject prob outside [0, 1]."""
        with pytest.raises(ValueError):
            dbinom(1, 10, 1.5)
