"""Test rsfus."""
import numpy as np
import pytest
from moirais.fn.rsfus import rsfus


def test_rsfus_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsfus(pixels=pixels, n=40)
    assert r.value is not None


def test_rsfus_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsfus(pixels=pixels, n=40)
    assert r.name
