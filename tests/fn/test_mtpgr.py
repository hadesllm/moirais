"""Test mtpgr."""
import numpy as np
import pytest
from morie.fn.mtpgr import mtpgr


def test_mtpgr_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtpgr(trajectory=traj, n=25)
    assert r.value is not None


def test_mtpgr_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtpgr(trajectory=traj, n=25)
    assert r.name
