"""Tests for moirais.fn.mtsqr -- Matrix square root."""

import numpy as np
from moirais.fn.mtsqr import matrix_sqrt, mtsqr
from moirais.fn._containers import DescriptiveResult


class TestMtsqr:
    def test_alias(self):
        assert mtsqr is matrix_sqrt

    def test_identity(self):
        r = matrix_sqrt(np.eye(3))
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["matrix"], np.eye(3), atol=1e-8)

    def test_diagonal(self):
        A = np.diag([4.0, 9.0])
        r = matrix_sqrt(A)
        S = r.extra["matrix"]
        np.testing.assert_allclose(S @ S, A, atol=1e-6)
