"""Tests for moirais.fn.sigplt -- generic signal plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.sigplt import sigplt


class TestSigPlt:
    def test_basic(self):
        x = np.sin(np.linspace(0, 2 * np.pi, 500))
        result = sigplt(x, fs=100)
        assert result.name == "signal_plot"
        assert result.value == 500
        assert result.extra["figure"] is not None
        plt.close(result.extra["figure"])

    def test_custom_labels(self):
        x = np.random.default_rng(42).standard_normal(200)
        result = sigplt(x, fs=50, title="Test", xlabel="t", ylabel="V")
        assert result.extra["figure"] is not None
        plt.close(result.extra["figure"])
