"""Tests for morie.fn.mlsmu — ML SMACOF unfolding."""
import numpy as np
import pytest

from morie.fn.mlsmu import mlsmu


def test_mlsmu_smoke():
    D_u = np.random.default_rng(42).random((4, 3)) + 0.5
    r = mlsmu(D_u, n_dims=1, max_iter=20, n_restarts=1)
    assert "respondent_coords" in r.extra


def test_cheatsheet():
    from morie.fn.mlsmu import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
