"""Tests for morie.fn.holo_s -- scatter plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
pd = pytest.importorskip("pandas")

from morie.fn.holo_s import holo_scatter


class TestHoloScatter:
    def test_returns_figure(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        fig = holo_scatter(df, "a", "b")
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_with_hue(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7], "g": ["x", "x", "y", "y"]})
        fig = holo_scatter(df, "a", "b", hue="g")
        assert fig is not None
        plt.close(fig)
