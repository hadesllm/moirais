"""Tests for moirais.fn.holo_h -- histogram."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.holo_h import holo_hist


class TestHoloHist:
    def test_returns_figure(self):
        data = {"x": np.random.default_rng(0).normal(size=100)}
        fig = holo_hist(data, "x")
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_custom_bins(self):
        data = {"x": np.arange(50, dtype=float)}
        fig = holo_hist(data, "x", bins=10)
        assert fig is not None
        plt.close(fig)
