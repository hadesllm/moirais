"""Tests for moirais.fn.ppoi — Poisson CDF."""

import numpy as np
import pytest

from moirais.fn.ppoi import ppois


class TestPpois:
    """Tests for ppois()."""

    def test_at_zero(self):
        """ppois(0, lambda_=1) = e^{-1} ~ 0.3679."""
        assert ppois(0, lambda_=1.0) == pytest.approx(0.3679, abs=1e-3)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [ppois(x, lambda_=2.0) for x in [0, 1, 2, 5, 10]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_type(self):
        """Returns float for scalar input."""
        result = ppois(3, lambda_=2.0)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_lambda(self):
        """Should reject lambda_ <= 0."""
        with pytest.raises(ValueError):
            ppois(0, lambda_=0)
