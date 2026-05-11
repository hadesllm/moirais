"""Tests for morie.fn.mtexp -- Matrix exponential."""

import numpy as np
from morie.fn.mtexp import matrix_exp, mtexp
from morie.fn._containers import DescriptiveResult


class TestMtexp:
    def test_alias(self):
        assert mtexp is matrix_exp

    def test_zero(self):
        r = matrix_exp(np.zeros((3, 3)))
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["matrix"], np.eye(3), atol=1e-10)

    def test_diagonal(self):
        A = np.diag([1.0, 2.0])
        r = matrix_exp(A)
        expected = np.diag([np.e, np.e**2])
        np.testing.assert_allclose(r.extra["matrix"], expected, atol=1e-6)
