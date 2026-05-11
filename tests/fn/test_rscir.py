"""Test rscir."""
import numpy as np
import pytest
from morie.fn.rscir import rscir


def test_rscir_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rscir(pixels=pixels, n=40)
    assert r.value is not None


def test_rscir_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rscir(pixels=pixels, n=40)
    assert r.name
