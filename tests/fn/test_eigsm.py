"""Tests for moirais.fn.eigsm -- symmetric eigenvalue decomposition."""

import numpy as np
import pytest
from moirais.fn.eigsm import eigen_symmetric, eigsm
from moirais.fn._containers import DescriptiveResult


class TestEigsm:
    def test_alias(self):
        assert eigsm is eigen_symmetric

    def test_identity(self):
        r = eigen_symmetric(np.eye(3))
        assert abs(r.value - 1.0) < 1e-12
        np.testing.assert_allclose(r.extra["eigenvalues"], [1, 1, 1], atol=1e-12)

    def test_symmetric(self):
        A = np.array([[2, 1], [1, 2]], dtype=float)
        r = eigen_symmetric(A)
        np.testing.assert_allclose(sorted(r.extra["eigenvalues"]), [1, 3], atol=1e-10)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            eigen_symmetric(np.ones((2, 3)))
