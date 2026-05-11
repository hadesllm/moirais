"""Tests for morie.fn.mtlog -- Matrix logarithm."""

import numpy as np
from morie.fn.mtlog import matrix_log, mtlog
from morie.fn._containers import DescriptiveResult


class TestMtlog:
    def test_alias(self):
        assert mtlog is matrix_log

    def test_identity(self):
        r = matrix_log(np.eye(3))
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["matrix"], np.zeros((3, 3)), atol=1e-8)

    def test_diagonal(self):
        A = np.diag([np.e, np.e**2])
        r = matrix_log(A)
        np.testing.assert_allclose(np.diag(r.extra["matrix"]), [1.0, 2.0], atol=0.5)
