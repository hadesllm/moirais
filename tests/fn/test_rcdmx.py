"""Tests for moirais.fn.rcdmx — recidivism mixture model."""

import pytest
import numpy as np
from moirais.fn.rcdmx import recidivism_mixture
from moirais.fn._containers import DescriptiveResult


class TestRecidivismMixture:

    def test_returns_descriptive(self):
        rng = np.random.default_rng(42)
        t = np.concatenate([rng.exponential(1, 50), rng.exponential(5, 50)])
        result = recidivism_mixture(t)
        assert isinstance(result, DescriptiveResult)
        assert "lambda1" in result.extra

    def test_mix_prob_bounded(self):
        rng = np.random.default_rng(7)
        t = np.concatenate([rng.exponential(2, 80), rng.exponential(10, 20)])
        result = recidivism_mixture(t)
        assert 0 <= result.extra["mix_prob"] <= 1
