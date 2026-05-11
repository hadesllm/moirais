"""Tests for morie.fn.imfplt -- IMF decomposition plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from morie.fn.imfplt import imfplt


class TestImfPlt:
    def test_basic(self):
        rng = np.random.default_rng(42)
        imfs = [rng.standard_normal(500) for _ in range(4)]
        result = imfplt(imfs, fs=100)
        assert result.name == "imf_plot"
        assert result.value == 4
        assert result.extra["n_imfs"] == 4
        plt.close(result.extra["figure"])

    def test_single_imf(self):
        imfs = [np.sin(np.linspace(0, 2 * np.pi, 200))]
        result = imfplt(imfs, fs=50)
        assert result.value == 1
        plt.close(result.extra["figure"])
