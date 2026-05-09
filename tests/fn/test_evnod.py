"""Test even_odd_decompose (evnod)."""
import numpy as np
import pytest

from moirais.fn.evnod import even_odd_decompose, evnod
from moirais.fn._containers import DescriptiveResult


class TestEvenOddDecompose:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = even_odd_decompose(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "even_odd_decompose"

    def test_reconstruction(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = even_odd_decompose(x)
        reconstructed = result.extra["even"] + result.extra["odd"]
        np.testing.assert_allclose(reconstructed, x)

    def test_even_symmetry(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = even_odd_decompose(x)
        even = result.extra["even"]
        np.testing.assert_allclose(even, even[::-1])

    def test_alias(self):
        assert evnod is even_odd_decompose
