"""Tests for moirais.fn.qbnm — binomial quantile function."""

import numpy as np
import pytest

from moirais.fn.qbnm import qbinom


class TestQbinom:
    """Tests for qbinom()."""

    def test_median_fair_coin(self):
        """qbinom(0.5, 10, 0.5) = 5."""
        result = qbinom(0.5, 10, 0.5)
        assert int(result) == 5

    def test_low_quantile(self):
        """qbinom(0.01, 10, 0.5) should be small integer."""
        result = qbinom(0.01, 10, 0.5)
        assert 0 <= int(result) <= 10

    def test_type(self):
        """Returns integer or integer-like."""
        result = qbinom(0.5, 10, 0.5)
        assert isinstance(result, (int, np.integer, float, np.floating))

    def test_raises_bad_prob(self):
        """Should reject prob outside [0, 1]."""
        with pytest.raises(ValueError):
            qbinom(0.5, 10, 2.0)
