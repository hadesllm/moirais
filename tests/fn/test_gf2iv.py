"""Test gf2_matrix_inv."""
import numpy as np
import pytest
from moirais.fn.gf2iv import gf2_matrix_inv
from moirais.fn._containers import DescriptiveResult


class TestGf2MatrixInv:
    def test_basic(self):
        I = np.eye(3, dtype=int)
        result = gf2_matrix_inv(a=I)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        I = np.eye(3, dtype=int)
        result = gf2_matrix_inv(a=I)
        assert "inverse" in result.extra

    def test_identity_inverse(self):
        I = np.eye(3, dtype=int)
        result = gf2_matrix_inv(a=I)
        r = np.asarray(result.extra["inverse"])
        np.testing.assert_array_equal(r, I)
