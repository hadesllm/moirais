"""Test rssrf."""
import numpy as np
import pytest
from moirais.fn.rssrf import rssrf


def test_rssrf_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssrf(pixels=pixels, n=40)
    assert r.value is not None


def test_rssrf_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssrf(pixels=pixels, n=40)
    assert r.name
