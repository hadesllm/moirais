"""Test rsndv."""
import numpy as np
import pytest
from morie.fn.rsndv import rsndv


def test_rsndv_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsndv(pixels=pixels, n=40)
    assert r.value is not None


def test_rsndv_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsndv(pixels=pixels, n=40)
    assert r.name
