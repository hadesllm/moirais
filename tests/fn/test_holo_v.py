"""Tests for moirais.fn.holo_v -- violin plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
pd = pytest.importorskip("pandas")

from moirais.fn.holo_v import holo_violin


class TestHoloViolin:
    def test_returns_figure(self):
        df = pd.DataFrame({"x": np.random.default_rng(0).normal(size=50)})
        fig = holo_violin(df, "x")
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_grouped(self):
        df = pd.DataFrame({"x": np.arange(20, dtype=float), "g": ["a"] * 10 + ["b"] * 10})
        fig = holo_violin(df, "x", group="g")
        assert fig is not None
        plt.close(fig)
