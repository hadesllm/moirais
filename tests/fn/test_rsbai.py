"""Test rsbai."""
import numpy as np
import pytest
from morie.fn.rsbai import rsbai


def test_rsbai_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsbai(pixels=pixels, n=40)
    assert r.value is not None


def test_rsbai_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsbai(pixels=pixels, n=40)
    assert r.name
