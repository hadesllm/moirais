"""Tests for moirais.fn.spcplt -- power spectrum plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.spcplt import spcplt


class TestSpcPlt:
    def test_basic_log(self):
        freqs = np.linspace(0, 50, 256)
        psd = np.abs(np.random.default_rng(42).standard_normal(256)) + 0.01
        result = spcplt(psd, freqs)
        assert result.name == "spectrum_plot"
        assert result.value == 256
        assert result.extra["log_scale"] is True
        plt.close(result.extra["figure"])

    def test_linear(self):
        freqs = np.linspace(0, 100, 128)
        psd = np.ones(128)
        result = spcplt(psd, freqs, log_scale=False)
        assert result.extra["log_scale"] is False
        plt.close(result.extra["figure"])
