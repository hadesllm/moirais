"""Test rsfcv."""
import numpy as np
import pytest
from morie.fn.rsfcv import rsfcv


def test_rsfcv_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsfcv(pixels=pixels, n=40)
    assert r.value is not None


def test_rsfcv_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsfcv(pixels=pixels, n=40)
    assert r.name
