"""Test abmsm."""
import numpy as np
import pytest
from morie.fn.abmsm import abmsm


def test_abmsm_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = abmsm(points=pts, n=40)
    assert r.value is not None


def test_abmsm_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = abmsm(points=pts, n=40)
    assert r.name
