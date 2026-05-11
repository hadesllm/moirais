"""Tests for morie.fn.mmds — Metric MDS."""
import numpy as np
from morie.fn.mmds import mmds


def test_mmds_basic():
    D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
    r = mmds(D, n_dims=2)
    assert r.coordinates.shape == (3, 2)
    assert r.stress >= 0


def test_mmds_identity():
    D = np.array([[0, 0.1, 0.2], [0.1, 0, 0.15], [0.2, 0.15, 0]])
    r = mmds(D, n_dims=1)
    assert r.stress < 1.0


def test_mmds_1d():
    pts = np.array([0, 1, 3, 6])
    D = np.abs(pts[:, None] - pts[None, :]).astype(float)
    r = mmds(D, n_dims=1)
    assert r.coordinates.shape == (4, 1)
    assert r.stress < 0.1
