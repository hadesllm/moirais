"""Test rskap."""
import numpy as np
import pytest
from morie.fn.rskap import rskap


def test_rskap_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rskap(pixels=pixels, n=40)
    assert r.value is not None


def test_rskap_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rskap(pixels=pixels, n=40)
    assert r.name
