"""Tests for moirais.fn.holo_m -- mosaic plot."""

import pytest

plt = pytest.importorskip("matplotlib.pyplot")
pd = pytest.importorskip("pandas")

from moirais.fn.holo_m import holo_mosaic


class TestHoloMosaic:
    def test_returns_figure(self):
        df = pd.DataFrame({
            "row": ["A", "A", "B", "B", "B", "C"],
            "col": ["x", "y", "x", "x", "y", "x"],
        })
        fig = holo_mosaic(df, "row", "col")
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_on_axes(self):
        df = pd.DataFrame({"r": ["a", "b"], "c": ["x", "y"]})
        fig0, ax = plt.subplots()
        fig = holo_mosaic(df, "r", "c", ax=ax)
        assert fig is fig0
        plt.close(fig)
