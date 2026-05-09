"""Tests for moirais.fn.mtxop -- matrix function."""

import numpy as np
from moirais.fn.mtxop import matrix_function, mtxop
from moirais.fn._containers import DescriptiveResult


class TestMtxop:
    def test_alias(self):
        assert mtxop is matrix_function

    def test_exp_identity(self):
        A = np.eye(3)
        result = matrix_function(A, func="exp")
        assert isinstance(result, DescriptiveResult)
        R = np.array(result.value)
        assert np.allclose(R, np.eye(3) * np.e, atol=1e-8)

    def test_inv(self):
        A = np.diag([2.0, 4.0])
        result = matrix_function(A, func="inv")
        R = np.array(result.value)
        assert np.allclose(R, np.diag([0.5, 0.25]), atol=1e-8)
