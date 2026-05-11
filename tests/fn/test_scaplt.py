"""Tests for morie.fn.scaplt -- scalogram plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from morie.fn.scaplt import scaplt


class TestScaPlt:
    def test_basic(self):
        scales = np.arange(1, 33, dtype=float)
        coeffs = np.random.default_rng(42).standard_normal((32, 500))
        result = scaplt(coeffs, scales, fs=100)
        assert result.name == "scalogram_plot"
        assert result.value == 32
        assert result.extra["n_scales"] == 32
        plt.close(result.extra["figure"])


def test_cheatsheet():
    from morie.fn.scaplt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
