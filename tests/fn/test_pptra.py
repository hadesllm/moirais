"""Test pptra."""
import numpy as np
import pytest
from morie.fn.pptra import pptra


def test_pptra_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pptra(points=pts, n=30)
    assert r.value is not None


def test_pptra_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pptra(points=pts, n=30)
    assert r.name
