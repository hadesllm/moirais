"""Test ppmhr."""
import numpy as np
import pytest
from moirais.fn.ppmhr import ppmhr


def test_ppmhr_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmhr(points=pts, n=30)
    assert r.value is not None


def test_ppmhr_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmhr(points=pts, n=30)
    assert r.name
