"""Test mtodm."""
import numpy as np
import pytest
from morie.fn.mtodm import mtodm


def test_mtodm_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtodm(trajectory=traj, n=25)
    assert r.value is not None


def test_mtodm_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtodm(trajectory=traj, n=25)
    assert r.name
