"""Test rsend."""
import numpy as np
import pytest
from moirais.fn.rsend import rsend


def test_rsend_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsend(pixels=pixels, n=40)
    assert r.value is not None


def test_rsend_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsend(pixels=pixels, n=40)
    assert r.name
