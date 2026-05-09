"""Tests for moirais.fn.pexp — exponential CDF."""

import numpy as np
import pytest

from moirais.fn.pexp import pexp


class TestPexp:
    """Tests for pexp()."""

    def test_at_zero(self):
        """pexp(0) == 0."""
        assert pexp(0, rate=1.0) == pytest.approx(0.0, abs=1e-12)

    def test_at_inf(self):
        """pexp(inf) == 1."""
        assert pexp(np.inf, rate=1.0) == pytest.approx(1.0, abs=1e-12)

    def test_known_value(self):
        """pexp(1, rate=1) = 1 - e^{-1} ~ 0.6321."""
        expected = 1.0 - np.exp(-1.0)
        assert pexp(1, rate=1.0) == pytest.approx(expected, rel=1e-10)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [pexp(x, rate=1.0) for x in [0, 0.5, 1, 2, 5]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_upper_tail(self):
        """lower_tail=False gives 1 - CDF."""
        q = 2.0
        lower = pexp(q, rate=1.0)
        upper = pexp(q, rate=1.0, lower_tail=False)
        assert lower + upper == pytest.approx(1.0, abs=1e-12)

    def test_array_input(self):
        """Should handle array input."""
        q = np.array([0, 1, 2])
        result = pexp(q, rate=1.0)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_raises_on_nonpositive_rate(self):
        """Should reject rate <= 0."""
        with pytest.raises(ValueError):
            pexp(1, rate=0)
        with pytest.raises(ValueError):
            pexp(1, rate=-1)
