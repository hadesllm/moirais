"""Tests for morie.fn.rcdcm — recidivism competing risks."""

import pytest
import numpy as np
from morie.fn.rcdcm import recidivism_competing
from morie.fn._containers import DescriptiveResult


class TestRecidivismCompeting:

    def test_returns_descriptive(self):
        times = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        events = np.array([1, 0, 2, 1, 0, 1, 2, 0])
        result = recidivism_competing(times, events)
        assert isinstance(result, DescriptiveResult)
        assert "cif_type_1" in result.extra

    def test_cif_nondecreasing(self):
        rng = np.random.default_rng(42)
        times = rng.exponential(3, 50)
        events = rng.choice([0, 1, 2], 50, p=[0.3, 0.5, 0.2])
        result = recidivism_competing(times, events)
        cif1 = result.extra["cif_type_1"]
        for i in range(1, len(cif1)):
            assert cif1[i] >= cif1[i - 1] - 1e-10
