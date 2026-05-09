"""Tests for moirais.fn.holo_e -- effect size plot."""

import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.holo_e import holo_effect


class TestHoloEffect:
    def test_returns_figure(self):
        est = [0.3, 0.5, -0.1]
        cis = [(0.1, 0.5), (0.2, 0.8), (-0.4, 0.2)]
        fig = holo_effect(est, cis)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_with_labels(self):
        fig = holo_effect([0.2], [(0.0, 0.4)], labels=["ATE"])
        assert fig is not None
        plt.close(fig)
