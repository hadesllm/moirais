"""Tests for morie.fn.qnorm — normal quantile function."""

import numpy as np
import pytest

from morie.fn.qnorm import qnorm


class TestQnorm:
    """Tests for qnorm()."""

    def test_median(self):
        """qnorm(0.5) = 0.0 for standard normal."""
        assert qnorm(0.5) == pytest.approx(0.0, abs=1e-12)

    def test_975(self):
        """qnorm(0.975) ~ 1.96."""
        assert qnorm(0.975) == pytest.approx(1.96, abs=1e-2)

    def test_025(self):
        """qnorm(0.025) ~ -1.96."""
        assert qnorm(0.025) == pytest.approx(-1.96, abs=1e-2)

    def test_type(self):
        """Scalar input returns float."""
        result = qnorm(0.5)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_sd(self):
        """Should reject sd <= 0."""
        with pytest.raises(ValueError):
            qnorm(0.5, sd=0)
