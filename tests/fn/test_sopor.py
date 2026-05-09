"""Test sopor."""
import numpy as np
import pytest
from moirais.fn.sopor import sopor


def test_sopor_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = sopor(data=data, depth=depth, n=20)
    assert r.value is not None


def test_sopor_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = sopor(data=data, depth=depth, n=20)
    assert r.name
