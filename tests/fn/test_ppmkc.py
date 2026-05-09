"""Test ppmkc."""
import numpy as np
import pytest
from moirais.fn.ppmkc import ppmkc


def test_ppmkc_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmkc(points=pts, n=30)
    assert r.value is not None


def test_ppmkc_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmkc(points=pts, n=30)
    assert r.name
