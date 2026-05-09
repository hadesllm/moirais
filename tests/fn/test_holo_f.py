"""Tests for moirais.fn.holo_f -- forest plot."""

import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.holo_f import holo_forest


class TestHoloForest:
    def test_returns_figure(self):
        effects = [0.5, 0.3, 0.8, 0.1]
        ses = [0.1, 0.15, 0.2, 0.05]
        fig = holo_forest(effects, ses)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_with_labels(self):
        fig = holo_forest([0.2, 0.4], [0.1, 0.1], labels=["A", "B"])
        assert fig is not None
        plt.close(fig)
