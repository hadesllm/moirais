"""Tests for morie.fn.eegplt -- EEG montage plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from morie.fn.eegplt import eegplt


class TestEegPlt:
    def test_basic_multichannel(self):
        channels = np.random.default_rng(42).standard_normal((8, 2048))
        result = eegplt(channels, fs=256)
        assert result.name == "eeg_montage"
        assert result.value == 8
        assert result.extra["figure"] is not None
        assert result.extra["n_channels"] == 8
        plt.close(result.extra["figure"])

    def test_single_channel(self):
        ch = np.random.default_rng(42).standard_normal(1024)
        result = eegplt(ch, fs=256)
        assert result.value == 1
        plt.close(result.extra["figure"])

    def test_with_duration(self):
        channels = np.random.default_rng(42).standard_normal((4, 2560))
        result = eegplt(channels, fs=256, duration=5.0)
        assert result.extra["figure"] is not None
        plt.close(result.extra["figure"])
