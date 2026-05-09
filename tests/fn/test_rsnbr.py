"""Test rsnbr."""
import numpy as np
import pytest
from moirais.fn.rsnbr import rsnbr


def test_rsnbr_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsnbr(pixels=pixels, n=40)
    assert r.value is not None


def test_rsnbr_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsnbr(pixels=pixels, n=40)
    assert r.name
