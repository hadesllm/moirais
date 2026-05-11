"""Test rslst."""
import numpy as np
import pytest
from morie.fn.rslst import rslst


def test_rslst_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rslst(pixels=pixels, n=40)
    assert r.value is not None


def test_rslst_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rslst(pixels=pixels, n=40)
    assert r.name
