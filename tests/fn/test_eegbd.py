"""Tests for moirais.fn.eegbd -- EEG band decomposition plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
pytest.importorskip("scipy")

from moirais.fn.eegbd import eegbd


class TestEegBd:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(2560)
        result = eegbd(x, fs=256)
        assert result.name == "eeg_bands"
        assert result.value == 5
        assert result.extra["figure"] is not None
        plt.close(result.extra["figure"])


def test_cheatsheet():
    from moirais.fn.eegbd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
