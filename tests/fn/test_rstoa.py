"""Test rstoa."""
import numpy as np
import pytest
from morie.fn.rstoa import rstoa


def test_rstoa_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rstoa(pixels=pixels, n=40)
    assert r.value is not None


def test_rstoa_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rstoa(pixels=pixels, n=40)
    assert r.name
