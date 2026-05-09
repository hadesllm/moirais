"""Tests for moirais.fn.dgeom — geometric PMF."""

import numpy as np
import pytest

from moirais.fn.dgeom import dgeom


class TestDgeom:
    """Tests for dgeom()."""

    def test_at_zero(self):
        """dgeom(0, prob=0.5) = 0.5 (first success on first trial)."""
        assert dgeom(0, prob=0.5) == pytest.approx(0.5, abs=1e-12)

    def test_at_one(self):
        """dgeom(1, prob=0.5) = 0.25 (one failure then success)."""
        assert dgeom(1, prob=0.5) == pytest.approx(0.25, abs=1e-12)

    def test_type(self):
        """Scalar input returns float."""
        result = dgeom(2, prob=0.3)
        assert isinstance(result, (float, np.floating))

    def test_raises_bad_prob(self):
        """Should reject prob not in (0, 1]."""
        with pytest.raises(ValueError):
            dgeom(0, prob=0)
