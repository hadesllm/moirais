"""Test rspca."""
import numpy as np
import pytest
from morie.fn.rspca import rspca


def test_rspca_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rspca(pixels=pixels, n=40)
    assert r.value is not None


def test_rspca_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rspca(pixels=pixels, n=40)
    assert r.name
