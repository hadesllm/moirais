"""Tests for moirais.fn.skwrp -- Levy flight."""

import numpy as np
from moirais.fn.skwrp import levy_flight, skwrp
from moirais.fn._containers import DescriptiveResult


class TestSkwrp:
    def test_alias(self):
        assert skwrp is levy_flight

    def test_shape(self):
        r = levy_flight(n_steps=100, dim=2)
        assert isinstance(r, DescriptiveResult)
        assert r.value.shape == (101, 2)

    def test_starts_at_origin(self):
        r = levy_flight(n_steps=50, dim=3)
        np.testing.assert_array_equal(r.value[0], [0, 0, 0])
