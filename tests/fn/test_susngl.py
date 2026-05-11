"""Tests for morie.fn.susngl -- single-use risk."""

import pytest
from morie.fn.susngl import single_use_risk


class TestSingleUseRisk:
    def test_basic(self):
        res = single_use_risk([0.02, 0.06, 0.10, 0.20, 0.35])
        assert res.name == "single_use_risk"
        assert res.extra["risk_categories_pct"]["life_threatening"] == pytest.approx(20.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            single_use_risk([])
