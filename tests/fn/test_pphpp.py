"""Test pphpp."""
import numpy as np
import pytest
from morie.fn.pphpp import pphpp


def test_pphpp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pphpp(points=pts, n=30)
    assert r.value is not None


def test_pphpp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pphpp(points=pts, n=30)
    assert r.name
