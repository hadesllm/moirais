"""Tests for moirais.fn.pf — F-distribution CDF."""

import numpy as np
import pytest

from moirais.fn.pf import pf


class TestPf:
    """Tests for pf()."""

    def test_at_zero(self):
        """pf(0, df1, df2) == 0."""
        assert pf(0, 5, 5) == pytest.approx(0.0, abs=1e-12)

    def test_at_inf(self):
        """pf(inf, df1, df2) == 1."""
        assert pf(np.inf, 5, 5) == pytest.approx(1.0, abs=1e-12)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [pf(x, 3, 4) for x in [0.5, 1.0, 2.0, 5.0]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_upper_tail(self):
        """lower_tail=False gives 1 - CDF."""
        q = 2.0
        lower = pf(q, 5, 5)
        upper = pf(q, 5, 5, lower_tail=False)
        assert lower + upper == pytest.approx(1.0, abs=1e-12)

    def test_array_input(self):
        """Should handle array input."""
        q = np.array([0.5, 1.0, 2.0])
        result = pf(q, 5, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_raises_on_nonpositive_df(self):
        """Should reject dfn <= 0 or dfd <= 0."""
        with pytest.raises(ValueError):
            pf(1, 0, 5)
        with pytest.raises(ValueError):
            pf(1, 5, -1)
