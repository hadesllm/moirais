"""Tests for morie.fn.qf — F-distribution quantile."""

import numpy as np
import pytest

from morie.fn.qf import qf
from morie.fn.pf import pf


class TestQf:
    """Tests for qf()."""

    def test_round_trip(self):
        """qf(pf(x)) should approximate x."""
        x = 2.5
        p = pf(x, 5, 5)
        x_hat = qf(p, 5, 5)
        assert x_hat == pytest.approx(x, rel=1e-10)

    def test_median(self):
        """qf(0.5, df1, df2) is the median."""
        med = qf(0.5, 10, 10)
        assert med > 0
        # Median of F(10,10) is close to 1
        assert abs(med - 1.0) < 0.1

    def test_extremes(self):
        """qf(0) == 0 and qf(1) == inf."""
        assert qf(0, 5, 5) == pytest.approx(0.0, abs=1e-12)
        assert np.isinf(qf(1, 5, 5))

    def test_array_input(self):
        """Should handle array input."""
        p = np.array([0.1, 0.5, 0.9])
        result = qf(p, 5, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert np.all(np.diff(result) > 0)  # monotone

    def test_raises_on_nonpositive_df(self):
        """Should reject dfn <= 0 or dfd <= 0."""
        with pytest.raises(ValueError):
            qf(0.5, 0, 5)
        with pytest.raises(ValueError):
            qf(0.5, 5, -1)
