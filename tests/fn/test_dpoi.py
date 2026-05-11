"""Tests for morie.fn.dpoi — Poisson PMF."""

import numpy as np
import pytest

from morie.fn.dpoi import dpois


class TestDpois:
    """Tests for dpois()."""

    def test_at_zero(self):
        """dpois(0, lambda_=1) = e^{-1} ~ 0.3679."""
        assert dpois(0, lambda_=1.0) == pytest.approx(0.3679, abs=1e-3)

    def test_at_one(self):
        """dpois(1, lambda_=1) = e^{-1} ~ 0.3679."""
        assert dpois(1, lambda_=1.0) == pytest.approx(0.3679, abs=1e-3)

    def test_type(self):
        """Returns float for scalar input."""
        result = dpois(2, lambda_=3.0)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_lambda(self):
        """Should reject lambda_ <= 0."""
        with pytest.raises(ValueError):
            dpois(0, lambda_=0)
