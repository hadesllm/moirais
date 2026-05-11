"""Tests for morie.fn.rchisq — chi-squared random variates."""

import numpy as np
import pytest

from morie.fn.rchisq import rchisq


class TestRchisq:
    """Tests for rchisq()."""

    def test_length(self):
        """Output length matches requested n."""
        result = rchisq(200, df=4)
        assert len(result) == 200

    def test_mean_near_df(self):
        """For chi-squared, E[X] = df."""
        df = 8
        result = rchisq(100_000, df=df, seed=0)
        assert abs(np.mean(result) - df) < 0.1

    def test_all_nonnegative(self):
        """Chi-squared variates are always >= 0."""
        result = rchisq(10_000, df=2, seed=5)
        assert np.all(result >= 0)

    def test_reproducible(self):
        """Same seed produces identical output."""
        a = rchisq(50, df=3, seed=99)
        b = rchisq(50, df=3, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_raises_on_nonpositive_df(self):
        """Should reject df <= 0."""
        with pytest.raises(ValueError):
            rchisq(10, df=0)
        with pytest.raises(ValueError):
            rchisq(10, df=-5)
