"""Tests for morie.fn.eta2 -- Eta-squared from ANOVA F-statistic."""

import pytest
from morie.fn.eta2 import eta_squared


class TestEtaSquared:
    def test_known_value(self):
        """eta2 = df_b * F / (df_b * F + df_w)."""
        # F=10, df_between=2, df_within=97
        expected = (2 * 10) / (2 * 10 + 97)  # 20/117 ~ 0.1709
        assert eta_squared(10, 2, 97) == pytest.approx(expected, abs=1e-10)

    def test_zero_f_gives_zero(self):
        """F=0 should yield eta2=0."""
        assert eta_squared(0, 2, 97) == pytest.approx(0.0)

    def test_raises_on_negative_f(self):
        """Negative F should raise ValueError."""
        with pytest.raises(ValueError):
            eta_squared(-1, 2, 97)
