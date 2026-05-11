"""Tests for morie.fn.holo_w -- funnel plot."""

import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from morie.fn.holo_w import holo_funnel


class TestHoloFunnel:
    def test_returns_figure(self):
        effects = [0.5, 0.3, 0.8, 0.1, 0.6]
        ses = [0.1, 0.15, 0.2, 0.05, 0.12]
        fig = holo_funnel(effects, ses)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_on_axes(self):
        fig0, ax = plt.subplots()
        fig = holo_funnel([0.3, 0.5], [0.1, 0.2], ax=ax)
        assert fig is fig0
        plt.close(fig)
