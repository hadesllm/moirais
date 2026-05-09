"""Tests for moirais.fn.qq_ — QQ plot data."""
import numpy as np
import pytest
from moirais.fn.qq_ import qq_plot_data


class TestQQPlot:
    def test_uniform_pvals(self):
        pv = np.linspace(0.01, 1.0, 100)
        res = qq_plot_data(pv)
        assert res.value == pytest.approx(1.0, abs=0.3)

    def test_invalid_pval_raises(self):
        with pytest.raises(ValueError):
            qq_plot_data(np.array([0.0, 0.5]))
