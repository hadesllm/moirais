"""Test rsacc."""
import numpy as np
import pytest
from moirais.fn.rsacc import rsacc


def test_rsacc_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsacc(pixels=pixels, n=40)
    assert r.value is not None


def test_rsacc_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsacc(pixels=pixels, n=40)
    assert r.name
