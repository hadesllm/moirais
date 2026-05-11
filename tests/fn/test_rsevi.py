"""Test rsevi."""
import numpy as np
import pytest
from morie.fn.rsevi import rsevi


def test_rsevi_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsevi(pixels=pixels, n=40)
    assert r.value is not None


def test_rsevi_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsevi(pixels=pixels, n=40)
    assert r.name
