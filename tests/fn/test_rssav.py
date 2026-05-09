"""Test rssav."""
import numpy as np
import pytest
from moirais.fn.rssav import rssav


def test_rssav_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssav(pixels=pixels, n=40)
    assert r.value is not None


def test_rssav_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssav(pixels=pixels, n=40)
    assert r.name
