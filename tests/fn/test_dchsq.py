"""Tests for moirais.fn.dchsq — chi-squared PDF."""

import numpy as np
import pytest

from moirais.fn.dchsq import dchisq


class TestDchisq:
    """Tests for dchisq()."""

    def test_df1_at_1(self):
        """dchisq(1, df=1) is a positive float."""
        result = dchisq(1, df=1)
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_nonnegative(self):
        """PDF is non-negative for x >= 0."""
        x = np.linspace(0.01, 10, 50)
        result = dchisq(x, df=3)
        assert np.all(result >= 0)

    def test_raises_nonpositive_df(self):
        """Should reject df <= 0."""
        with pytest.raises(ValueError):
            dchisq(1, df=0)
