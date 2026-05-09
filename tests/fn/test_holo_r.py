"""Tests for moirais.fn.holo_r -- residual plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.holo_r import holo_resid


class TestHoloResid:
    def test_returns_figure(self):
        rng = np.random.default_rng(0)
        fitted = rng.normal(size=50)
        residuals = rng.normal(0, 0.5, size=50)
        fig = holo_resid(fitted, residuals)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_on_existing_axes(self):
        fig0, ax = plt.subplots()
        fig = holo_resid([1, 2, 3], [0.1, -0.2, 0.05], ax=ax)
        assert fig is fig0
        plt.close(fig)
