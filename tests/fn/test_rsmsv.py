"""Test rsmsv."""
import numpy as np
import pytest
from moirais.fn.rsmsv import rsmsv


def test_rsmsv_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmsv(pixels=pixels, n=40)
    assert r.value is not None


def test_rsmsv_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmsv(pixels=pixels, n=40)
    assert r.name
