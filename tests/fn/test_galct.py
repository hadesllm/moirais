"""Tests for morie.fn.galct -- resource depletion model."""

import numpy as np
from morie.fn.galct import depletion_model, galct
from morie.fn._containers import DescriptiveResult


class TestGalct:
    def test_alias(self):
        assert galct is depletion_model

    def test_exponential(self):
        t = np.arange(10, dtype=float)
        stock = 100 * np.exp(-0.2 * t)
        r = depletion_model(t, stock, model="exponential")
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value["rate"] - 0.2) < 0.05
        assert r.value["r_squared"] > 0.99

    def test_linear(self):
        t = np.arange(10, dtype=float)
        stock = 100 - 5 * t
        stock = np.maximum(stock, 0.1)
        r = depletion_model(t, stock, model="linear")
        assert r.value["time_to_exhaustion"] < 30
