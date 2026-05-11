"""Test percs."""
import numpy as np
import pytest
from morie.fn.percs import percs


def test_percs_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = percs(points=pts, n=40)
    assert r.value is not None


def test_percs_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = percs(points=pts, n=40)
    assert r.name
