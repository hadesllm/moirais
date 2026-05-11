"""Tests for morie.fn.tajd -- Tajima's D."""

import pytest
from morie.fn.tajd import tajimas_d


class TestTajimasD:
    def test_neutral_near_zero(self):
        """When pi = S/a1 (neutral expectation), D should be near 0."""
        n = 10
        S = 20
        a1 = sum(1.0 / i for i in range(1, n))
        pi = S / a1
        res = tajimas_d(S=S, n=n, pi=pi)
        assert res.name == "Tajima_D"
        assert res.statistic == pytest.approx(0.0, abs=1e-6)

    def test_positive_D(self):
        """Excess diversity (pi >> S/a1) gives positive D."""
        n = 20
        S = 5
        a1 = sum(1.0 / i for i in range(1, n))
        pi = S / a1 + 5.0
        res = tajimas_d(S=S, n=n, pi=pi)
        assert res.statistic > 0

    def test_negative_D(self):
        """Deficit of diversity (pi << S/a1) gives negative D."""
        n = 20
        S = 30
        a1 = sum(1.0 / i for i in range(1, n))
        pi = S / a1 - 5.0
        if pi < 0:
            pi = 0.0
        res = tajimas_d(S=S, n=n, pi=pi)
        assert res.statistic < 0

    def test_too_few_sequences(self):
        with pytest.raises(ValueError):
            tajimas_d(S=5, n=2, pi=3.0)
