"""Tests for morie.fn.holo_d -- DAG diagram."""

import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from morie.fn.holo_d import holo_dag


class TestHoloDag:
    def test_returns_figure(self):
        edges = [("X", "Y"), ("Z", "X"), ("Z", "Y")]
        fig = holo_dag(edges)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_single_edge(self):
        fig = holo_dag([("A", "B")])
        assert fig is not None
        plt.close(fig)
