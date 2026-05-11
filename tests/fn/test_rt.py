"""Tests for morie.fn.rt — Student t random variates."""

import numpy as np
import pytest

from morie.fn.rt import rt


class TestRt:
    """Tests for rt()."""

    def test_length(self):
        """Output length matches requested n."""
        result = rt(100, df=5)
        assert len(result) == 100

    def test_mean_near_zero(self):
        """For large n, sample mean should be near 0."""
        result = rt(50_000, df=10, seed=0)
        assert abs(np.mean(result)) < 0.05

    def test_variance_near_theoretical(self):
        """For df>2, Var(T) = df/(df-2)."""
        df = 10
        result = rt(100_000, df=df, seed=7)
        expected_var = df / (df - 2)
        assert abs(np.var(result, ddof=1) - expected_var) < 0.15

    def test_reproducible(self):
        """Same seed produces identical output."""
        a = rt(50, df=3, seed=99)
        b = rt(50, df=3, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_raises_on_nonpositive_df(self):
        """Should reject df <= 0."""
        with pytest.raises(ValueError):
            rt(10, df=0)
        with pytest.raises(ValueError):
            rt(10, df=-1)
