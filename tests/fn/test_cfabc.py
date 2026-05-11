"""Tests for cfabc -- BIC."""
from morie.fn.cfabc import cfa_bic
from morie.fn._containers import ESRes


class TestCfaBic:
    def test_basic(self):
        result = cfa_bic(-100.0, 10, 200)
        assert isinstance(result, ESRes)
        assert result.n == 200

    def test_larger_n_penalises_more(self):
        r1 = cfa_bic(-100.0, 10, 100)
        r2 = cfa_bic(-100.0, 10, 1000)
        assert r2.estimate > r1.estimate
