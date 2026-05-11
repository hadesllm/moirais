"""Tests for morie.fn.pt — Student's t CDF."""

import numpy as np
import pytest

from morie.fn.pt import pt


class TestPt:
    """Tests for pt()."""

    def test_at_zero(self):
        """pt(0, df=10) = 0.5 (symmetric about 0)."""
        assert pt(0, df=10) == pytest.approx(0.5, abs=1e-12)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [pt(x, df=5) for x in [-3, -1, 0, 1, 3]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_type(self):
        """Scalar input returns float."""
        result = pt(0.0, df=10)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_df(self):
        """Should reject df <= 0."""
        with pytest.raises(ValueError):
            pt(0, df=0)
