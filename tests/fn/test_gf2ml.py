"""Test gf2_matrix_mul."""
import numpy as np
import pytest
from morie.fn.gf2ml import gf2_matrix_mul
from morie.fn._containers import DescriptiveResult


class TestGf2MatrixMul:
    def test_basic(self):
        I = np.eye(3, dtype=int)
        M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int)
        result = gf2_matrix_mul(a=I, b=M)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        I = np.eye(3, dtype=int)
        M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int)
        result = gf2_matrix_mul(a=I, b=M)
        assert "result" in result.extra

    def test_identity_mul(self):
        I = np.eye(3, dtype=int)
        M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int)
        result = gf2_matrix_mul(a=I, b=M)
        r = np.asarray(result.extra["result"])
        np.testing.assert_array_equal(r, M)
