"""Test rsemi."""
import numpy as np
import pytest
from moirais.fn.rsemi import rsemi


def test_rsemi_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsemi(pixels=pixels, n=40)
    assert r.value is not None


def test_rsemi_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsemi(pixels=pixels, n=40)
    assert r.name
