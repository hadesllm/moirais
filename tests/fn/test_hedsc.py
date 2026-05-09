"""Tests for moirais.fn.hedsc -- discount rate."""

import pytest
from moirais.fn.hedsc import discount_rate


class TestDiscountRate:
    def test_basic(self):
        res = discount_rate(values=[100, 100, 100], years=[0, 1, 2], rate=0.05)
        assert res.value < 300.0
        assert res.extra["undiscounted_total"] == pytest.approx(300.0)

    def test_zero_rate(self):
        res = discount_rate([100, 100], [0, 1], rate=0.0)
        assert res.value == pytest.approx(200.0)
