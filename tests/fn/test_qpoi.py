"""Tests for morie.fn.qpoi — Poisson quantile function."""

import numpy as np
import pytest

from morie.fn.qpoi import qpois


class TestQpois:
    """Tests for qpois()."""

    def test_median_lambda1(self):
        """qpois(0.5, lambda_=1) = 1."""
        result = qpois(0.5, lambda_=1.0)
        assert int(result) == 1

    def test_high_quantile(self):
        """qpois(0.99, lambda_=1) should be a small positive int."""
        result = qpois(0.99, lambda_=1.0)
        assert int(result) >= 1

    def test_type(self):
        """Returns integer-like."""
        result = qpois(0.5, lambda_=5.0)
        assert isinstance(result, (int, np.integer, float, np.floating))

    def test_raises_nonpositive_lambda(self):
        """Should reject lambda_ <= 0."""
        with pytest.raises(ValueError):
            qpois(0.5, lambda_=0)
