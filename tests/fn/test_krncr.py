"""Tests for morie.fn.krncr -- Kronecker product."""

import numpy as np
from morie.fn.krncr import kronecker, krncr
from morie.fn._containers import DescriptiveResult


class TestKrncr:
    def test_alias(self):
        assert krncr is kronecker

    def test_identity(self):
        r = kronecker(np.eye(2), np.eye(2))
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["matrix"], np.eye(4))
        assert r.value == 16

    def test_shape(self):
        A = np.ones((2, 3))
        B = np.ones((4, 5))
        r = kronecker(A, B)
        assert r.extra["shape"] == (8, 15)
