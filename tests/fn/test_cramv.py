"""Tests for moirais.fn.cramv -- Cramer's V."""

import numpy as np
import pytest
from moirais.fn.cramv import cramers_v


class TestCramersV:
    def test_independent_table(self):
        """Independent rows/columns give V near 0."""
        # Perfectly proportional table
        table = [[50, 50], [50, 50]]
        v = cramers_v(table)
        assert isinstance(v, float)
        assert v == pytest.approx(0.0, abs=0.01)

    def test_strong_association(self):
        """Diagonal table gives V close to 1."""
        table = [[100, 0], [0, 100]]
        v = cramers_v(table)
        assert v > 0.95

    def test_raises_on_1d(self):
        """1-D input should raise ValueError."""
        with pytest.raises(ValueError):
            cramers_v([10, 20, 30])
