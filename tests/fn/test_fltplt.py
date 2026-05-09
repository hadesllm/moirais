"""Tests for moirais.fn.fltplt -- filter I/O plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.fltplt import fltplt


class TestFltPlt:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x_in = rng.standard_normal(500)
        x_out = x_in * 0.5
        result = fltplt(x_in, x_out, fs=100)
        assert result.name == "filter_io_plot"
        assert result.value == 500
        assert result.extra["figure"] is not None
        plt.close(result.extra["figure"])

    def test_different_lengths(self):
        x_in = np.ones(300)
        x_out = np.ones(250)
        result = fltplt(x_in, x_out, fs=50)
        assert result.extra["figure"] is not None
        plt.close(result.extra["figure"])
