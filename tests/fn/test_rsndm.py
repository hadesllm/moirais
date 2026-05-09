"""Test rsndm."""
import numpy as np
import pytest
from moirais.fn.rsndm import rsndm


def test_rsndm_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsndm(pixels=pixels, n=40)
    assert r.value is not None


def test_rsndm_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsndm(pixels=pixels, n=40)
    assert r.name
