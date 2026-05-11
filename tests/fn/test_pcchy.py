"""Tests for morie.fn.pcchy -- Cauchy CDF."""

import numpy as np
import pytest
from morie.fn.pcchy import pcchy


class TestPcchy:
    def test_at_zero(self):
        """pcchy(0) = 0.5 for standard Cauchy."""
        assert pcchy(0) == pytest.approx(0.5, abs=1e-10)

    def test_upper_tail(self):
        """pcchy(0, lower_tail=False) = 0.5."""
        assert pcchy(0, lower_tail=False) == pytest.approx(0.5, abs=1e-10)

    def test_monotone(self):
        """CDF is monotone increasing."""
        vals = pcchy(np.array([-10, -1, 0, 1, 10]))
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_raises_bad_scale(self):
        with pytest.raises(ValueError):
            pcchy(0, scale=-1)
