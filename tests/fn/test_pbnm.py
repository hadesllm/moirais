"""Tests for morie.fn.pbnm — binomial CDF."""

import numpy as np
import pytest

from morie.fn.pbnm import pbinom


class TestPbinom:
    """Tests for pbinom()."""

    def test_median(self):
        """pbinom(5, 10, 0.5) ~ 0.6230."""
        assert pbinom(5, 10, 0.5) == pytest.approx(0.6230, abs=1e-3)

    def test_all_trials(self):
        """pbinom(10, 10, 0.5) = 1.0."""
        assert pbinom(10, 10, 0.5) == pytest.approx(1.0, abs=1e-12)

    def test_type(self):
        """Returns float for scalar input."""
        result = pbinom(3, 10, 0.5)
        assert isinstance(result, (float, np.floating))

    def test_raises_bad_prob(self):
        """Should reject prob outside [0, 1]."""
        with pytest.raises(ValueError):
            pbinom(1, 10, -0.1)
