"""Tests for moirais.fn.holo_p -- pair plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
pd = pytest.importorskip("pandas")

from moirais.fn.holo_p import holo_pair


class TestHoloPair:
    def test_returns_figure(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
        fig = holo_pair(df)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_subset_cols(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        fig = holo_pair(df, cols=["a", "c"])
        assert fig is not None
        plt.close(fig)
