"""Test rsndb."""
import numpy as np
import pytest
from morie.fn.rsndb import rsndb


def test_rsndb_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsndb(pixels=pixels, n=40)
    assert r.value is not None


def test_rsndb_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsndb(pixels=pixels, n=40)
    assert r.name
