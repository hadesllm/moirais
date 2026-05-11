"""Test reproj."""
import numpy as np
import pytest
from morie.fn.reproj import reproj


def test_reproj_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = reproj(coords=coords, n=20)
    assert r.value is not None


def test_reproj_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = reproj(coords=coords, n=20)
    assert r.name
