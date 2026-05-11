"""Test pplgc."""
import numpy as np
import pytest
from morie.fn.pplgc import pplgc


def test_pplgc_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pplgc(points=pts, n=30)
    assert r.value is not None


def test_pplgc_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pplgc(points=pts, n=30)
    assert r.name
