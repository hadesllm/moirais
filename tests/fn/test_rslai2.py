"""Test rslai2."""
import numpy as np
import pytest
from morie.fn.rslai2 import rslai2


def test_rslai2_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rslai2(pixels=pixels, n=40)
    assert r.value is not None


def test_rslai2_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rslai2(pixels=pixels, n=40)
    assert r.name
