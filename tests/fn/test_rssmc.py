"""Test rssmc."""
import numpy as np
import pytest
from morie.fn.rssmc import rssmc


def test_rssmc_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssmc(pixels=pixels, n=40)
    assert r.value is not None


def test_rssmc_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rssmc(pixels=pixels, n=40)
    assert r.name
