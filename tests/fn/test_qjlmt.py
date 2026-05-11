"""Tests for morie.fn.qjlmt — QJL projection matrix."""

import numpy as np
import pytest

from morie.fn.qjlmt import qjl_matrix


class TestQjlMatrix:

    def test_shape(self):
        res = qjl_matrix(128, 32)
        assert res.extra["matrix"].shape == (128, 32)

    def test_rademacher_values(self):
        res = qjl_matrix(64, 16)
        R = res.extra["matrix"]
        scaled = R * np.sqrt(16)
        assert np.allclose(np.abs(scaled), 1.0, atol=1e-10)

    def test_reproducible(self):
        r1 = qjl_matrix(32, 8, seed=0).extra["matrix"]
        r2 = qjl_matrix(32, 8, seed=0).extra["matrix"]
        assert np.allclose(r1, r2)
