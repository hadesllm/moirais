"""Test bkz_reduce."""
import numpy as np
import pytest
from moirais.fn.bkz import bkz_reduce
from moirais.fn._containers import DescriptiveResult


class TestBkzReduce:
    def test_basic(self):
        basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)
        result = bkz_reduce(basis=basis, block_size=2)
        assert result.extra["dimension"] is not None

    def test_output_type(self):
        basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)
        result = bkz_reduce(basis=basis, block_size=2)
        assert isinstance(result, DescriptiveResult)

    def test_reduced_basis_in_extra(self):
        basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)
        result = bkz_reduce(basis=basis, block_size=2)
        assert "reduced_basis" in result.extra
