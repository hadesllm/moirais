"""Test rsfpr."""
import numpy as np
import pytest
from moirais.fn.rsfpr import rsfpr


def test_rsfpr_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsfpr(pixels=pixels, n=40)
    assert r.value is not None


def test_rsfpr_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsfpr(pixels=pixels, n=40)
    assert r.name
