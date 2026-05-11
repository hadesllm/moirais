"""Tests for morie.fn.qlnrm — lognormal quantile function."""

import numpy as np
import pytest

from morie.fn.qlnrm import qlnrm


class TestQlnrm:
    """Tests for qlnrm()."""

    def test_median(self):
        """qlnrm(0.5, meanlog=0, sdlog=1) = exp(0) = 1.0."""
        assert qlnrm(0.5, meanlog=0, sdlog=1) == pytest.approx(1.0, abs=1e-6)

    def test_positive(self):
        """All quantiles are positive."""
        for p in [0.1, 0.5, 0.9]:
            assert qlnrm(p, meanlog=0, sdlog=1) > 0

    def test_type(self):
        """Scalar input returns float."""
        result = qlnrm(0.5, meanlog=0, sdlog=1)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_sdlog(self):
        """Should reject sdlog <= 0."""
        with pytest.raises(ValueError):
            qlnrm(0.5, sdlog=0)
