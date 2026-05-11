"""Test mtmlr."""
import numpy as np
import pytest
from morie.fn.mtmlr import mtmlr


def test_mtmlr_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtmlr(trajectory=traj, n=25)
    assert r.value is not None


def test_mtmlr_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtmlr(trajectory=traj, n=25)
    assert r.name
