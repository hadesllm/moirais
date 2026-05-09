"""Tests for moirais.fn.qchsq — chi-squared quantile function."""

import numpy as np
import pytest

from moirais.fn.qchsq import qchisq


class TestQchisq:
    """Tests for qchisq()."""

    def test_95_df1(self):
        """qchisq(0.95, df=1) ~ 3.841."""
        assert qchisq(0.95, df=1) == pytest.approx(3.841, abs=1e-2)

    def test_95_df2(self):
        """qchisq(0.95, df=2) ~ 5.991."""
        assert qchisq(0.95, df=2) == pytest.approx(5.991, abs=1e-2)

    def test_type(self):
        """Scalar input returns float."""
        result = qchisq(0.5, df=5)
        assert isinstance(result, (float, np.floating))

    def test_raises_nonpositive_df(self):
        """Should reject df <= 0."""
        with pytest.raises(ValueError):
            qchisq(0.5, df=0)
