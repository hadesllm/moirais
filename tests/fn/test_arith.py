"""Tests for moirais.fn.arith — arithmetic coding."""

import numpy as np
import pytest

from moirais.fn.arith import arith


class TestArith:
    def test_roundtrip(self):
        pmf = np.array([0.5, 0.3, 0.2])
        msg = [0, 1, 2, 0, 1]
        result = arith(msg, pmf)
        assert result["decoded"] == msg

    def test_binary_roundtrip(self):
        pmf = np.array([0.7, 0.3])
        msg = [0, 0, 1, 0, 1, 1]
        result = arith(msg, pmf)
        assert result["decoded"] == msg

    def test_code_in_interval(self):
        pmf = np.array([0.5, 0.5])
        msg = [0, 1, 0]
        result = arith(msg, pmf)
        lo, hi = result["interval"]
        assert lo <= result["code_value"] <= hi

    def test_bits_required_positive(self):
        pmf = np.array([0.4, 0.3, 0.2, 0.1])
        msg = [0, 1, 2, 3]
        result = arith(msg, pmf)
        assert result["bits_required"] > 0

    def test_custom_symbols(self):
        pmf = np.array([0.6, 0.4])
        msg = ["A", "B", "A"]
        result = arith(msg, pmf, symbols=["A", "B"])
        assert result["decoded"] == msg

    def test_unknown_symbol_error(self):
        with pytest.raises(ValueError):
            arith([0, 5], np.array([0.5, 0.5]))
