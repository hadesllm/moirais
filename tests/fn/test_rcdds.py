"""Tests for moirais.fn.rcdds — recidivism desistance."""

import pytest
import numpy as np
from moirais.fn.rcdds import recidivism_desistance
from moirais.fn._containers import DescriptiveResult


class TestRecidivismDesistance:

    def test_returns_descriptive(self):
        t = np.array([1, 2, 3, 4, 5, 10, 15, 20])
        result = recidivism_desistance(t)
        assert isinstance(result, DescriptiveResult)
        assert "desistance_prob" in result.extra

    def test_desistance_decreasing(self):
        rng = np.random.default_rng(42)
        t = rng.exponential(3, 100)
        result = recidivism_desistance(t)
        d = result.extra["desistance_prob"]
        for i in range(1, len(d)):
            assert d[i] <= d[i - 1] + 1e-10
