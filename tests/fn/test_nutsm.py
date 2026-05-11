"""Test nutsm."""
import numpy as np
import pytest
from morie.fn.nutsm import nutsm


def test_nutsm_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = nutsm(points=pts, n=40)
    assert r.value is not None


def test_nutsm_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = nutsm(points=pts, n=40)
    assert r.name
