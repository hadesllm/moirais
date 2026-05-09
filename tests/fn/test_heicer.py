"""Tests for moirais.fn.heicer -- ICER."""

import pytest
from moirais.fn.heicer import incremental_cer


class TestICER:
    def test_basic(self):
        res = incremental_cer(cost_new=5000, cost_old=3000, effect_new=5, effect_old=3)
        assert res.estimate == pytest.approx(1000.0)

    def test_dominant(self):
        res = incremental_cer(3000, 5000, 5, 3)
        assert res.extra["quadrant"] == "SE_dominant"
