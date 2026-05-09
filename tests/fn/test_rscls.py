"""Test rscls."""
import numpy as np
import pytest
from moirais.fn.rscls import rscls


def test_rscls_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rscls(pixels=pixels, n=40)
    assert r.value is not None


def test_rscls_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rscls(pixels=pixels, n=40)
    assert r.name
