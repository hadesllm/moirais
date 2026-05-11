"""Test lwe_key_exchange."""
import numpy as np
import pytest
from morie.fn.lweke import lwe_key_exchange
from morie.fn._containers import CryptoResult


class TestLweKeyExchange:
    def test_basic(self):
        for _ in range(10):
            result = lwe_key_exchange(n=64, q=3329, sigma=3.2)
            if result.success:
                break
        assert result.success is True

    def test_output_type(self):
        result = lwe_key_exchange(n=8, q=97)
        assert isinstance(result, CryptoResult)

    def test_extra_keys(self):
        result = lwe_key_exchange(n=8, q=97)
        assert "alice_key" in result.extra
        assert "bob_key" in result.extra
