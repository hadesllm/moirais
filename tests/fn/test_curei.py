"""Tests for moirais.fn.curei — Mixture cure model."""

import numpy as np
import pytest

from moirais.fn.curei import cure_model


class TestCureModel:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        cured = rng.uniform(size=n) < 0.3
        times = np.where(cured, rng.uniform(5, 20, n), rng.weibull(2, n) * 3)
        events = np.where(cured, 0, 1).astype(int)
        events[rng.uniform(size=n) < 0.1] = 0
        res = cure_model(times, events)
        assert 0 < res.extra["cure_fraction"] < 1

    def test_weibull_shape_positive(self):
        rng = np.random.default_rng(42)
        times = rng.weibull(1.5, 100) * 5
        events = np.ones(100, dtype=int)
        events[:20] = 0
        res = cure_model(times, events)
        assert res.extra["weibull_shape"] > 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            cure_model([1, 2], [1])
