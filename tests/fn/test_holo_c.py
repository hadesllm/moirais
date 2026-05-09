"""Tests for moirais.fn.holo_c -- correlation heatmap."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
pd = pytest.importorskip("pandas")

from moirais.fn.holo_c import holo_corr


class TestHoloCorr:
    def test_returns_figure(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [1, 3, 2, 4]})
        fig = holo_corr(df)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_spearman(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})
        fig = holo_corr(df, method="spearman")
        assert fig is not None
        plt.close(fig)
