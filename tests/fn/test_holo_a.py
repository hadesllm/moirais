"""Tests for moirais.fn.holo_a -- ACF/PACF plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.holo_a import holo_acf


class TestHoloAcf:
    def test_acf_returns_figure(self):
        vals = np.random.default_rng(0).normal(size=100)
        fig = holo_acf(vals, nlags=20)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_pacf_returns_figure(self):
        vals = np.random.default_rng(1).normal(size=100)
        fig = holo_acf(vals, pacf=True, nlags=15)
        assert fig is not None
        plt.close(fig)
