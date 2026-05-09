"""Test gf2m_arithmetic."""
import numpy as np
import pytest
from moirais.fn.gf2m import gf2m_arithmetic
from moirais.fn._containers import DescriptiveResult


class TestGf2mArithmetic:
    def test_basic(self):
        result = gf2m_arithmetic(op="add", a=3, b=5, m=8)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        result = gf2m_arithmetic(op="add", a=3, b=5, m=8)
        assert "result" in result.extra

    def test_add_xor(self):
        result = gf2m_arithmetic(op="add", a=3, b=5, m=8)
        assert result.extra["result"] == 6

    def test_mul(self):
        result = gf2m_arithmetic(op="mul", a=3, b=5, m=8)
        assert "result" in result.extra

    def test_inv(self):
        result = gf2m_arithmetic(op="inv", a=3, b=0, m=8)
        assert "result" in result.extra
