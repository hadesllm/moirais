"""Test pttsp."""
import numpy as np
import pytest
from moirais.fn.pttsp import pttsp


def test_pttsp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = pttsp(points=pts, n=40)
    assert r.value is not None


def test_pttsp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = pttsp(points=pts, n=40)
    assert r.name
