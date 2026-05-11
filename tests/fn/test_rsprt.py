"""Tests for morie.fn.rsprt -- respiratory rate estimation."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from morie.fn.rsprt import rsprt


class TestRspRt:
    def test_known_rate(self):
        fs = 100.0
        duration = 30.0
        t = np.arange(int(fs * duration)) / fs
        resp_rate_hz = 15.0 / 60.0
        x = np.sin(2 * np.pi * resp_rate_hz * t)
        result = rsprt(x, fs=fs)
        assert result.name == "respiratory_rate"
        assert 10 < result.value < 20

    def test_flat_signal(self):
        x = np.zeros(1000)
        result = rsprt(x, fs=100)
        assert result.value == 0.0
