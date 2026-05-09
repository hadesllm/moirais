"""Tests for moirais.fn.gtime — Generation time distribution."""

import numpy as np
import pytest

from moirais.fn.gtime import generation_time


class TestGenerationTime:
    def test_gamma_fit(self):
        rng = np.random.default_rng(42)
        gt = rng.gamma(3, 2, 100)
        res = generation_time(gt, distribution="gamma")
        assert res.value > 0
        assert res.extra["n"] == 100

    def test_lognormal(self):
        rng = np.random.default_rng(42)
        gt = rng.lognormal(1, 0.5, 100)
        res = generation_time(gt, distribution="lognormal")
        assert res.extra["aic"] is not None

    def test_too_few(self):
        with pytest.raises(ValueError):
            generation_time([1.0, 2.0])
