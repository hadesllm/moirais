"""Tests for morie.fn.sgmplt -- spectrogram plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from morie.fn.sgmplt import sgmplt


class TestSgmPlt:
    def test_basic(self):
        Sxx = np.abs(np.random.default_rng(42).standard_normal((64, 100))) + 0.01
        result = sgmplt(Sxx, fs=256)
        assert result.name == "spectrogram_plot"
        assert result.value == 64
        assert result.extra["shape"] == (64, 100)
        plt.close(result.extra["figure"])

    def test_with_axes(self):
        Sxx = np.abs(np.random.default_rng(42).standard_normal((32, 50))) + 0.01
        t = np.linspace(0, 1, 50)
        f = np.linspace(0, 128, 32)
        result = sgmplt(Sxx, fs=256, t=t, f=f)
        assert result.extra["figure"] is not None
        plt.close(result.extra["figure"])
