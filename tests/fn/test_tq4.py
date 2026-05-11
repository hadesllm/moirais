"""Tests for morie.fn.tq4 — 4-bit TurboQuant."""

import numpy as np
import pytest

from morie.fn.tq4 import turboquant_4bit


class TestTurboquant4bit:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = turboquant_4bit(x)
        assert res.extra["bits"] == 4

    def test_target_compression(self):
        x = np.random.default_rng(0).standard_normal(32)
        res = turboquant_4bit(x)
        assert res.extra["target_compression"] == 7.6
