"""Tests for moirais.fn.pqnrm — PolarQuant unit-sphere projection."""

import numpy as np
import pytest

from moirais.fn.pqnrm import polar_normalize


class TestPolarNormalize:

    def test_unit_norm(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = polar_normalize(x)
        assert res.extra["unit_check"] == pytest.approx(1.0, abs=1e-10)

    def test_returns_original_norm(self):
        x = np.array([3.0, 4.0])
        res = polar_normalize(x)
        assert res.value == pytest.approx(5.0)

    def test_zero_vector(self):
        x = np.zeros(10)
        res = polar_normalize(x)
        assert res.value == 0.0
