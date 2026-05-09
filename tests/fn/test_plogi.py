"""Tests for moirais.fn.plogi -- Logistic CDF."""

import numpy as np
import pytest
from moirais.fn.plogi import plogi


class TestPlogi:
    def test_at_zero(self):
        """plogi(0) = 0.5 for standard logistic."""
        assert plogi(0) == pytest.approx(0.5, abs=1e-10)

    def test_upper_tail(self):
        """plogi(0, lower_tail=False) = 0.5."""
        assert plogi(0, lower_tail=False) == pytest.approx(0.5, abs=1e-10)

    def test_monotone(self):
        """CDF is monotone increasing."""
        vals = plogi(np.array([-10, -1, 0, 1, 10]))
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_raises_bad_scale(self):
        with pytest.raises(ValueError):
            plogi(0, scale=-1)
