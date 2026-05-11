"""Test mtktz."""
import numpy as np
import pytest
from morie.fn.mtktz import mtktz


def test_mtktz_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtktz(trajectory=traj, n=25)
    assert r.value is not None


def test_mtktz_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtktz(trajectory=traj, n=25)
    assert r.name
