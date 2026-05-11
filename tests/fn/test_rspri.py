"""Test rspri."""
import numpy as np
import pytest
from morie.fn.rspri import rspri


def test_rspri_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rspri(pixels=pixels, n=40)
    assert r.value is not None


def test_rspri_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rspri(pixels=pixels, n=40)
    assert r.name
