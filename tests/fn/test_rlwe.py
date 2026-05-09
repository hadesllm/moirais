"""Test rlwe_keygen."""
import numpy as np
import pytest
from moirais.fn.rlwe import rlwe_keygen
from moirais.fn._containers import CryptoResult


class TestRlweKeygen:
    def test_basic(self):
        result = rlwe_keygen(n=16, q=97)
        assert result.success is True

    def test_output_type(self):
        result = rlwe_keygen(n=16, q=97)
        assert isinstance(result, CryptoResult)

    def test_extra_polynomials(self):
        result = rlwe_keygen(n=16, q=97)
        for key in ("a", "b", "s", "e"):
            assert key in result.extra
