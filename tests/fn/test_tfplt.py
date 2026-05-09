"""Tests for moirais.fn.tfplt -- time-frequency distribution plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from moirais.fn.tfplt import tfplt


class TestTfPlt:
    def test_basic(self):
        t = np.linspace(0, 1, 100)
        f = np.linspace(0, 50, 64)
        tfd_data = np.random.default_rng(42).standard_normal((64, 100))
        result = tfplt(tfd_data, t, f)
        assert result.name == "tfd_plot"
        assert result.value == 64
        assert result.extra["shape"] == (64, 100)
        plt.close(result.extra["figure"])


def test_cheatsheet():
    from moirais.fn.tfplt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
