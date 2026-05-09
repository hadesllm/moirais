"""Test rsmnd."""
import numpy as np
import pytest
from moirais.fn.rsmnd import rsmnd


def test_rsmnd_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmnd(pixels=pixels, n=40)
    assert r.value is not None


def test_rsmnd_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmnd(pixels=pixels, n=40)
    assert r.name
