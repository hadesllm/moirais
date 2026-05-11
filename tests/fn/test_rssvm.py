"""Test rssvm."""
import numpy as np
import pytest
from morie.fn.rssvm import rssvm


def test_rssvm_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssvm(pixels=pixels, n=40)
    assert r.value is not None


def test_rssvm_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssvm(pixels=pixels, n=40)
    assert r.name
