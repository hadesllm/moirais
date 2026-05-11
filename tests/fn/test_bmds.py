"""Tests for morie.fn.bmds — Bayesian multidimensional scaling."""
import numpy as np
import pytest

from morie.fn.bmds import bmds


def test_bmds_smoke():
    D = np.array([[0, 1, 2, 3], [1, 0, 1.5, 2.5], [2, 1.5, 0, 1], [3, 2.5, 1, 0]])
    r = bmds(D, n_dims=1, n_samples=50, burn_in=10)
    assert "coordinate_mean" in r.extra


def test_cheatsheet():
    from morie.fn.bmds import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
