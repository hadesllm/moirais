"""Tests for moirais.fn.dbet — beta PDF."""

import numpy as np
import pytest

from moirais.fn.dbet import dbeta


class TestDbeta:
    """Tests for dbeta()."""

    def test_uniform(self):
        """dbeta(0.5, alpha=1, beta=1) = 1.0 (uniform on [0,1])."""
        assert dbeta(0.5, alpha=1, beta=1) == pytest.approx(1.0, abs=1e-12)

    def test_symmetric(self):
        """dbeta(0.5, 2, 2) is symmetric peak."""
        assert dbeta(0.3, 2, 2) == pytest.approx(dbeta(0.7, 2, 2), abs=1e-12)

    def test_type(self):
        """Scalar input returns float."""
        result = dbeta(0.5, 2, 3)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_alpha(self):
        """Should reject alpha <= 0."""
        with pytest.raises(ValueError):
            dbeta(0.5, alpha=0, beta=1)
