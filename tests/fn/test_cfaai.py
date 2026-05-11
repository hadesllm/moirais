"""Tests for cfaai -- AIC."""
from morie.fn.cfaai import cfa_aic
from morie.fn._containers import ESRes


class TestCfaAic:
    def test_basic(self):
        result = cfa_aic(-100.0, 10)
        assert isinstance(result, ESRes)
        assert result.estimate == 220.0

    def test_more_params_higher(self):
        r1 = cfa_aic(-100.0, 10)
        r2 = cfa_aic(-100.0, 20)
        assert r2.estimate > r1.estimate
