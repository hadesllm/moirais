"""Test filsm."""
import numpy as np
import pytest
from moirais.fn.filsm import filsm


def test_filsm_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = filsm(points=pts, n=40)
    assert r.value is not None


def test_filsm_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = filsm(points=pts, n=40)
    assert r.name
