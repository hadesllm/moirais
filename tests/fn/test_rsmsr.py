"""Test rsmsr."""
import numpy as np
import pytest
from morie.fn.rsmsr import rsmsr


def test_rsmsr_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmsr(pixels=pixels, n=40)
    assert r.value is not None


def test_rsmsr_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmsr(pixels=pixels, n=40)
    assert r.name
