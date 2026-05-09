"""Tests for moirais.fn.arplt -- AR model poles plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.arplt import arplt


class TestArPlt:
    def test_basic(self):
        coeffs = np.array([0.5, -0.3])
        result = arplt(coeffs, fs=100)
        assert result.name == "ar_poles_plot"
        assert result.value == 2
        assert result.extra["order"] == 2
        plt.close(result.extra["figure"])

    def test_higher_order(self):
        coeffs = np.array([0.8, -0.4, 0.2, -0.1])
        result = arplt(coeffs, fs=256)
        assert result.value == 4
        plt.close(result.extra["figure"])
