"""Test rsrei."""
import numpy as np
import pytest
from morie.fn.rsrei import rsrei


def test_rsrei_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsrei(pixels=pixels, n=40)
    assert r.value is not None


def test_rsrei_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsrei(pixels=pixels, n=40)
    assert r.name
