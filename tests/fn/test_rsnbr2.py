"""Test rsnbr2."""
import numpy as np
import pytest
from morie.fn.rsnbr2 import rsnbr2


def test_rsnbr2_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsnbr2(pixels=pixels, n=40)
    assert r.value is not None


def test_rsnbr2_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsnbr2(pixels=pixels, n=40)
    assert r.name
