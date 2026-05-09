"""Tests for moirais.fn.dt — Student's t PDF."""

import numpy as np
import pytest

from moirais.fn.dt import dt


class TestDt:
    """Tests for dt()."""

    def test_at_zero_df1(self):
        """dt(0, df=1) = 1/pi ~ 0.3183 (Cauchy)."""
        assert dt(0, df=1) == pytest.approx(0.3183, abs=1e-3)

    def test_at_zero_df10(self):
        """dt(0, df=10) is float and positive."""
        result = dt(0, df=10)
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_symmetry(self):
        """t-distribution is symmetric about 0."""
        assert dt(-2.0, df=5) == pytest.approx(dt(2.0, df=5), abs=1e-12)

    def test_raises_nonpositive_df(self):
        """Should reject df <= 0."""
        with pytest.raises(ValueError):
            dt(0, df=0)
        with pytest.raises(ValueError):
            dt(0, df=-1)
