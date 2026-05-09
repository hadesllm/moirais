"""Test ppcox."""
import numpy as np
import pytest
from moirais.fn.ppcox import ppcox


def test_ppcox_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppcox(points=pts, n=30)
    assert r.value is not None


def test_ppcox_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppcox(points=pts, n=30)
    assert r.name
