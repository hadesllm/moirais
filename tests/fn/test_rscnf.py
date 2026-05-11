"""Test rscnf."""
import numpy as np
import pytest
from morie.fn.rscnf import rscnf


def test_rscnf_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rscnf(pixels=pixels, n=40)
    assert r.value is not None


def test_rscnf_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rscnf(pixels=pixels, n=40)
    assert r.name
