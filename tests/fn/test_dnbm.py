"""Tests for morie.fn.dnbm — negative binomial PMF."""

import numpy as np
import pytest

from morie.fn.dnbm import dnbm


class TestDnbm:
    """Tests for dnbm()."""

    def test_at_zero(self):
        """dnbm(0, size=1, prob=0.5) = 0.5."""
        assert dnbm(0, size=1, prob=0.5) == pytest.approx(0.5, abs=1e-12)

    def test_geometric_case(self):
        """dnbm(1, size=1, prob=0.5) = 0.25 (geometric special case)."""
        assert dnbm(1, size=1, prob=0.5) == pytest.approx(0.25, abs=1e-12)

    def test_type(self):
        """Returns float for scalar input."""
        result = dnbm(2, size=3, prob=0.4)
        assert isinstance(result, (float, np.floating))

    def test_raises_bad_prob(self):
        """Should reject prob not in (0, 1]."""
        with pytest.raises(ValueError):
            dnbm(0, size=1, prob=0)
