"""Test rsusc."""
import numpy as np
import pytest
from morie.fn.rsusc import rsusc


def test_rsusc_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsusc(pixels=pixels, n=40)
    assert r.value is not None


def test_rsusc_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsusc(pixels=pixels, n=40)
    assert r.name
