"""Tests for morie.fn.pinv — pseudoinverse."""
import numpy as np
from morie.fn.pinv import pseudoinverse


class TestPseudoinverse:
    def test_identity(self):
        res = pseudoinverse(np.eye(3))
        np.testing.assert_allclose(res.extra["pinv"], np.eye(3), atol=1e-10)

    def test_rank_deficient(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        res = pseudoinverse(A)
        assert res.extra["rank"] == 1
