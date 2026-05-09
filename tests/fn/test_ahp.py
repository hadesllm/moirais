"""Tests for moirais.fn.ahp — AHP weights."""
import numpy as np
import pytest
from moirais.fn.ahp import ahp_weights


class TestAHP:
    def test_consistent_matrix(self):
        A = np.array([[1, 3, 5], [1/3, 1, 5/3], [1/5, 3/5, 1]])
        res = ahp_weights(A)
        assert res.extra["consistent"]
        np.testing.assert_allclose(res.extra["weights"].sum(), 1.0, atol=1e-6)

    def test_identity(self):
        A = np.eye(3)
        res = ahp_weights(A)
        np.testing.assert_allclose(res.extra["weights"], [1/3, 1/3, 1/3], atol=0.01)
