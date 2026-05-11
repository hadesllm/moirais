"""Test rsalb."""
import numpy as np
import pytest
from morie.fn.rsalb import rsalb


def test_rsalb_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsalb(pixels=pixels, n=40)
    assert r.value is not None


def test_rsalb_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsalb(pixels=pixels, n=40)
    assert r.name
