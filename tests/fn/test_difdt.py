"""Tests for difdt -- Delta plot DIF."""
import numpy as np
from moirais.fn.difdt import dif_delta_plot
from moirais.fn._containers import DIFResult


class TestDifDeltaPlot:
    def test_basic(self):
        p_r = np.array([0.7, 0.6, 0.8, 0.5, 0.9])
        p_f = np.array([0.7, 0.6, 0.8, 0.5, 0.9])
        result = dif_delta_plot(p_r, p_f)
        assert isinstance(result, DIFResult)
        assert result.method == "DeltaPlot"

    def test_flagged_with_diff(self):
        p_r = np.array([0.7, 0.6, 0.5])
        p_f = np.array([0.7, 0.1, 0.5])
        result = dif_delta_plot(p_r, p_f, threshold=0.5)
        assert "perp_distance" in result.items.columns
