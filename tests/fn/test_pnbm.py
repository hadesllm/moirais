"""Tests for moirais.fn.pnbm — negative binomial CDF."""

import numpy as np
import pytest

from moirais.fn.pnbm import pnbm


class TestPnbm:
    """Tests for pnbm()."""

    def test_at_zero(self):
        """pnbm(0, size=1, prob=0.5) = 0.5."""
        assert pnbm(0, size=1, prob=0.5) == pytest.approx(0.5, abs=1e-12)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [pnbm(q, size=2, prob=0.5) for q in [0, 1, 2, 5, 10]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_type(self):
        """Returns float for scalar input."""
        result = pnbm(3, size=2, prob=0.4)
        assert isinstance(result, (float, np.floating))

    def test_raises_bad_prob(self):
        """Should reject prob not in (0, 1]."""
        with pytest.raises(ValueError):
            pnbm(0, size=1, prob=0)
