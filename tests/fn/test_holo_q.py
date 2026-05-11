"""Tests for morie.fn.holo_q -- QQ plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
pytest.importorskip("scipy")

from morie.fn.holo_q import holo_qq


class TestHoloQQ:
    def test_returns_figure(self):
        data = {"z": np.random.default_rng(1).normal(size=100)}
        fig = holo_qq(data, "z")
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_array_input(self):
        arr = np.random.default_rng(2).normal(size=50)
        fig = holo_qq(arr, "values")
        assert fig is not None
        plt.close(fig)
