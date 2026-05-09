"""Test bmsim."""
import numpy as np
import pytest
from moirais.fn.bmsim import bmsim


def test_bmsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = bmsim(points=pts, n=40)
    assert r.value is not None


def test_bmsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = bmsim(points=pts, n=40)
    assert r.name
