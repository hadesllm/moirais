"""Tests for morie.fn.pchsq — chi-squared CDF."""

import numpy as np
import pytest

from morie.fn.pchsq import pchisq


class TestPchisq:
    """Tests for pchisq()."""

    def test_critical_value(self):
        """pchisq(3.841, df=1) ~ 0.95."""
        assert pchisq(3.841, df=1) == pytest.approx(0.95, abs=1e-2)

    def test_at_zero(self):
        """pchisq(0, df=1) = 0."""
        assert pchisq(0, df=1) == pytest.approx(0.0, abs=1e-12)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [pchisq(x, df=2) for x in [0, 1, 2, 5, 10]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_raises_nonpositive_df(self):
        """Should reject df <= 0."""
        with pytest.raises(ValueError):
            pchisq(1, df=0)
