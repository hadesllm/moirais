"""Test ppmh2."""
import numpy as np
import pytest
from morie.fn.ppmh2 import ppmh2


def test_ppmh2_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmh2(points=pts, n=30)
    assert r.value is not None


def test_ppmh2_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmh2(points=pts, n=30)
    assert r.name
