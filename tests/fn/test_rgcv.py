"""Tests for rgcv -- generalizability coefficient."""
from moirais.fn.rgcv import generalizability_coeff
from moirais.fn._containers import ESRes


class TestGeneralizability:
    def test_basic(self):
        vc = {"person": 0.8, "item": 0.1, "error": 0.1, "n_items": 5}
        result = generalizability_coeff(vc)
        assert isinstance(result, ESRes)
        assert 0 < result.estimate <= 1

    def test_d_coefficient(self):
        vc = {"person": 0.5, "item": 0.2, "interaction": 0.1, "error": 0.2, "n_items": 10}
        result = generalizability_coeff(vc)
        assert "D_coefficient" in result.extra
        assert result.extra["D_coefficient"] <= result.estimate
