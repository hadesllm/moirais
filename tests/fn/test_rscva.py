"""Test rscva."""
import numpy as np
import pytest
from morie.fn.rscva import rscva


def test_rscva_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rscva(pixels=pixels, n=40)
    assert r.value is not None


def test_rscva_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rscva(pixels=pixels, n=40)
    assert r.name
