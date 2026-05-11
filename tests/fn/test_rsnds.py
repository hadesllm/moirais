"""Test rsnds."""
import numpy as np
import pytest
from morie.fn.rsnds import rsnds


def test_rsnds_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsnds(pixels=pixels, n=40)
    assert r.value is not None


def test_rsnds_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsnds(pixels=pixels, n=40)
    assert r.name
