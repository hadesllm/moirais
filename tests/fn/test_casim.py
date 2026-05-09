"""Test casim."""
import numpy as np
import pytest
from moirais.fn.casim import casim


def test_casim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = casim(points=pts, n=40)
    assert r.value is not None


def test_casim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = casim(points=pts, n=40)
    assert r.name
