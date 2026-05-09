"""Test loss_curve_analysis."""
import numpy as np
from moirais.fn.lscur import loss_curve_analysis, lscur
from moirais.fn._containers import DescriptiveResult


class TestLossCurveAnalysis:
    def test_converging(self):
        losses = [3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35]
        result = loss_curve_analysis(losses, window=3)
        assert isinstance(result, DescriptiveResult)
        assert result.value == "converging"

    def test_diverging(self):
        losses = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]
        result = loss_curve_analysis(losses, window=3)
        assert result.value == "diverging"

    def test_plateau(self):
        losses = [1.0] * 20
        result = loss_curve_analysis(losses, window=3)
        assert result.value == "plateau"

    def test_alias(self):
        assert lscur is loss_curve_analysis
