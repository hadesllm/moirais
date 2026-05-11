"""Tests for morie.fn.pqenc — PolarQuant encoder."""

import numpy as np
import pytest

from morie.fn.pqenc import polarquant_encode


class TestPolarquantEncode:

    def test_magnitude(self):
        x = np.array([3.0, 4.0])
        res = polarquant_encode(x)
        assert res.value == pytest.approx(5.0)

    def test_direction_unit(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = polarquant_encode(x)
        norm = np.linalg.norm(res.extra["direction"])
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_zero_vector(self):
        x = np.zeros(10)
        res = polarquant_encode(x)
        assert res.value == 0.0
