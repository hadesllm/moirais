"""Tests for moirais.fn.dlnrm — lognormal PDF."""

import numpy as np
import pytest

from moirais.fn.dlnrm import dlnrm


class TestDlnrm:
    """Tests for dlnrm()."""

    def test_at_one(self):
        """dlnrm(1, meanlog=0, sdlog=1) = dnorm(0) ~ 0.3989."""
        assert dlnrm(1, meanlog=0, sdlog=1) == pytest.approx(0.3989, abs=1e-3)

    def test_zero_density(self):
        """dlnrm(0) should be 0 (lognormal has no mass at 0)."""
        assert dlnrm(0, meanlog=0, sdlog=1) == pytest.approx(0.0, abs=1e-12)

    def test_type(self):
        """Scalar input returns float."""
        result = dlnrm(2.0, meanlog=0, sdlog=1)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_sdlog(self):
        """Should reject sdlog <= 0."""
        with pytest.raises(ValueError):
            dlnrm(1, sdlog=0)
