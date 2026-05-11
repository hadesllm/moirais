"""Test rsgnd."""
import numpy as np
import pytest
from morie.fn.rsgnd import rsgnd


def test_rsgnd_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsgnd(pixels=pixels, n=40)
    assert r.value is not None


def test_rsgnd_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsgnd(pixels=pixels, n=40)
    assert r.name
