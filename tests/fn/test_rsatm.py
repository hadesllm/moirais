"""Test rsatm."""
import numpy as np
import pytest
from moirais.fn.rsatm import rsatm


def test_rsatm_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsatm(pixels=pixels, n=40)
    assert r.value is not None


def test_rsatm_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsatm(pixels=pixels, n=40)
    assert r.name
