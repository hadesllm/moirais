"""Tests for morie.fn.holo_k -- KM survival curve."""

import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from morie.fn.holo_k import holo_km


class TestHoloKM:
    def test_returns_figure(self):
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        events = [1, 0, 1, 1, 0, 1, 0, 1]
        fig = holo_km(times, events)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_with_groups(self):
        times = [1, 2, 3, 4, 5, 6]
        events = [1, 0, 1, 1, 0, 1]
        groups = ["a", "a", "a", "b", "b", "b"]
        fig = holo_km(times, events, groups=groups)
        assert fig is not None
        plt.close(fig)
