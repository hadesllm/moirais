"""Tests for moirais.fn.qbet — beta quantile function."""

import numpy as np
import pytest

from moirais.fn.qbet import qbeta


class TestQbeta:
    """Tests for qbeta()."""

    def test_uniform_median(self):
        """qbeta(0.5, alpha=1, beta=1) = 0.5."""
        assert qbeta(0.5, alpha=1, beta=1) == pytest.approx(0.5, abs=1e-12)

    def test_extreme_low(self):
        """qbeta(0.0, ...) should be near 0."""
        result = qbeta(0.01, alpha=2, beta=2)
        assert result >= 0.0
        assert result < 0.5

    def test_type(self):
        """Scalar input returns float."""
        result = qbeta(0.5, alpha=2, beta=3)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_alpha(self):
        """Should reject alpha <= 0."""
        with pytest.raises(ValueError):
            qbeta(0.5, alpha=-1, beta=1)
