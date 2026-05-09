"""Tests for moirais.fn.shkwv -- 1-D wave equation solver."""

import numpy as np
from moirais.fn.shkwv import wave_equation_1d, shkwv
from moirais.fn._containers import DescriptiveResult


class TestShkwv:
    def test_alias(self):
        assert shkwv is wave_equation_1d

    def test_basic_solve(self):
        r = wave_equation_1d(n_x=50, n_t=100)
        assert isinstance(r, DescriptiveResult)
        assert r.value.shape == (100, 50)

    def test_boundary_conditions(self):
        r = wave_equation_1d(n_x=50, n_t=100)
        np.testing.assert_allclose(r.value[:, 0], 0.0)
        np.testing.assert_allclose(r.value[:, -1], 0.0)
