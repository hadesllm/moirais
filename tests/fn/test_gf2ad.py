"""Test gf2_matrix_add."""
import numpy as np
import pytest
from morie.fn.gf2ad import gf2_matrix_add
from morie.fn._containers import DescriptiveResult


class TestGf2MatrixAdd:
    def test_basic(self):
        I = np.eye(3, dtype=int)
        result = gf2_matrix_add(a=I, b=I)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        I = np.eye(3, dtype=int)
        result = gf2_matrix_add(a=I, b=I)
        assert "result" in result.extra

    def test_self_add_is_zero(self):
        I = np.eye(3, dtype=int)
        result = gf2_matrix_add(a=I, b=I)
        r = np.asarray(result.extra["result"])
        np.testing.assert_array_equal(r, np.zeros((3, 3), dtype=int))
