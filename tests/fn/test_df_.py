"""Tests for moirais.fn.df_ — F-distribution PDF."""

import numpy as np
import pytest
from scipy.stats import f as f_dist

from moirais.fn.df_ import df_dist, df_


class TestDfDist:
    """Tests for df_dist() / df_()."""

    def test_known_value(self):
        """df_(1, 5, 5) should match scipy directly."""
        expected = float(f_dist.pdf(1, 5, 5))
        result = df_dist(1, 5, 5)
        assert abs(result - expected) < 1e-12

    def test_at_zero(self):
        """df_(0, 5, 5) should be 0 for dfn > 2."""
        result = df_dist(0, 5, 5)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_positive_x(self):
        """PDF should be positive for x > 0."""
        result = df_dist(2.0, 3, 4)
        assert result > 0

    def test_alias(self):
        """df_ should be the same function as df_dist."""
        assert df_(1, 5, 5) == df_dist(1, 5, 5)

    def test_array_input(self):
        """Should handle array input."""
        x = np.array([0.5, 1.0, 2.0])
        result = df_dist(x, 5, 5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_raises_on_nonpositive_df(self):
        """Should reject dfn <= 0 or dfd <= 0."""
        with pytest.raises(ValueError):
            df_dist(1, 0, 5)
        with pytest.raises(ValueError):
            df_dist(1, 5, -1)
