"""Tests for moirais.fn.pgeom — geometric CDF."""

import numpy as np
import pytest

from moirais.fn.pgeom import pgeom


class TestPgeom:
    """Tests for pgeom()."""

    def test_at_zero(self):
        """pgeom(0, prob=0.5) = 0.5."""
        assert pgeom(0, prob=0.5) == pytest.approx(0.5, abs=1e-12)

    def test_at_one(self):
        """pgeom(1, prob=0.5) = 0.75."""
        assert pgeom(1, prob=0.5) == pytest.approx(0.75, abs=1e-12)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [pgeom(q, prob=0.3) for q in [0, 1, 2, 5, 10]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_raises_bad_prob(self):
        """Should reject prob not in (0, 1]."""
        with pytest.raises(ValueError):
            pgeom(0, prob=0)
