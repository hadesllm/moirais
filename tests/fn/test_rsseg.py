"""Test rsseg."""
import numpy as np
import pytest
from morie.fn.rsseg import rsseg


def test_rsseg_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsseg(pixels=pixels, n=40)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert -1.0 <= r.value <= 1.0


def test_rsseg_extra():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsseg(pixels=pixels, n=40)
    assert isinstance(r.name, str) and len(r.name) > 0
    assert r.extra["n_pixels"] == 40
    assert r.extra["n_bands"] == 4
