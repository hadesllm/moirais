"""Test rsrtx."""
import numpy as np
import pytest
from morie.fn.rsrtx import rsrtx


def test_rsrtx_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsrtx(pixels=pixels, n=40)
    assert r.value is not None


def test_rsrtx_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsrtx(pixels=pixels, n=40)
    assert r.name
