"""Test markg."""
import numpy as np
import pytest
from morie.fn.markg import markg


def test_markg_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = markg(points=pts, n=40)
    assert r.value is not None


def test_markg_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = markg(points=pts, n=40)
    assert r.name
