"""Tests for morie.fn.gntme -- Generation time distribution."""

import pytest
import numpy as np
from morie.fn.gntme import generation_time


class TestGenerationTime:
    def test_gamma(self):
        rng = np.random.default_rng(42)
        data = rng.gamma(shape=4, scale=1.0, size=100)
        res = generation_time(data, distribution="gamma")
        assert res.measure == "generation_time"
        assert res.estimate > 0

    def test_lognormal(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(mean=1.0, sigma=0.5, size=100)
        res = generation_time(data, distribution="lognormal")
        assert res.estimate > 0

    def test_too_few(self):
        with pytest.raises(ValueError):
            generation_time([1.0])
