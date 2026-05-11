"""Test rsshd."""
import numpy as np
import pytest
from morie.fn.rsshd import rsshd


def test_rsshd_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsshd(pixels=pixels, n=40)
    assert r.value is not None


def test_rsshd_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsshd(pixels=pixels, n=40)
    assert r.name
