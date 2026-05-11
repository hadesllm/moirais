"""Test bgsim."""
import numpy as np
import pytest
from morie.fn.bgsim import bgsim


def test_bgsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = bgsim(points=pts, n=40)
    assert r.value is not None


def test_bgsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = bgsim(points=pts, n=40)
    assert r.name
