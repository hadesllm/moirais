"""Test ppmks."""
import numpy as np
import pytest
from moirais.fn.ppmks import ppmks


def test_ppmks_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmks(points=pts, n=30)
    assert r.value is not None


def test_ppmks_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmks(points=pts, n=30)
    assert r.name
