"""Tests for moirais.fn.holo_i -- ROC curve."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.holo_i import holo_roc


class TestHoloRoc:
    def test_returns_figure(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        y_score = rng.random(size=100)
        fig = holo_roc(y_true, y_score)
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_perfect_classifier(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        fig = holo_roc(y_true, y_score)
        assert fig is not None
        plt.close(fig)
