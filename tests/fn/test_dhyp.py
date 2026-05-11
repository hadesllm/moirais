"""Tests for morie.fn.dhyp — hypergeometric PMF."""

import numpy as np
import pytest

from morie.fn.dhyp import dhyp


class TestDhyp:
    """Tests for dhyp()."""

    def test_known_value(self):
        """dhyp(1, m=5, n=5, k=5) — known hypergeometric value.

        C(5,1)*C(5,4)/C(10,5) = 5*5/252 ~ 0.0992.
        """
        assert dhyp(1, m=5, n=5, k=5) == pytest.approx(0.0992, abs=1e-3)

    def test_extreme(self):
        """dhyp(0, m=5, n=5, k=5) = C(5,0)*C(5,5)/C(10,5) = 1/252 ~ 0.00397."""
        assert dhyp(0, m=5, n=5, k=5) == pytest.approx(1 / 252, abs=1e-4)

    def test_type(self):
        """Returns float for scalar input."""
        result = dhyp(2, m=10, n=10, k=5)
        assert isinstance(result, (float, np.floating))

    def test_raises_negative(self):
        """Should reject negative parameters."""
        with pytest.raises(ValueError):
            dhyp(1, m=-1, n=5, k=5)
