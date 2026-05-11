"""Tests for morie.fn.qt — Student's t quantile function."""

import numpy as np
import pytest

from morie.fn.qt import qt


class TestQt:
    """Tests for qt()."""

    def test_median(self):
        """qt(0.5, df=10) ~ 0.0."""
        assert qt(0.5, df=10) == pytest.approx(0.0, abs=1e-6)

    def test_975_large_df(self):
        """qt(0.975, df=1000) ~ 1.96 (approaches normal)."""
        assert qt(0.975, df=1000) == pytest.approx(1.96, abs=0.05)

    def test_type(self):
        """Scalar input returns float."""
        result = qt(0.5, df=10)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_df(self):
        """Should reject df <= 0."""
        with pytest.raises(ValueError):
            qt(0.5, df=0)
