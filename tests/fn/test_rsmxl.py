"""Test rsmxl."""
import numpy as np
import pytest
from moirais.fn.rsmxl import rsmxl


def test_rsmxl_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmxl(pixels=pixels, n=40)
    assert r.value is not None


def test_rsmxl_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmxl(pixels=pixels, n=40)
    assert r.name
