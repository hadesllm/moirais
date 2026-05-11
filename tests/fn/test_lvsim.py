"""Test lvsim."""
import numpy as np
import pytest
from morie.fn.lvsim import lvsim


def test_lvsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = lvsim(points=pts, n=40)
    assert r.value is not None


def test_lvsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = lvsim(points=pts, n=40)
    assert r.name
