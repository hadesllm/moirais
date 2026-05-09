"""Test poly_ring_op."""
import numpy as np
import pytest
from moirais.fn.plyrn import poly_ring_op
from moirais.fn._containers import DescriptiveResult


class TestPolyRingOp:
    def test_basic(self):
        result = poly_ring_op(op="add", a=[1, 2], b=[3, 4], q=7)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        result = poly_ring_op(op="add", a=[1, 2], b=[3, 4], q=7)
        assert "result" in result.extra

    def test_add_mod(self):
        result = poly_ring_op(op="add", a=[1, 2], b=[3, 4], q=7)
        r = result.extra["result"]
        assert list(r) == [4, 6]
