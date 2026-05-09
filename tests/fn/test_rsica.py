"""Test rsica."""
import numpy as np
import pytest
from moirais.fn.rsica import rsica


def test_rsica_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsica(pixels=pixels, n=40)
    assert r.value is not None


def test_rsica_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsica(pixels=pixels, n=40)
    assert r.name
