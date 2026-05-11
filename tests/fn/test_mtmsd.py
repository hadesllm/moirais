"""Test mtmsd."""
import numpy as np
import pytest
from morie.fn.mtmsd import mtmsd


def test_mtmsd_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtmsd(trajectory=traj, n=25)
    assert r.value is not None


def test_mtmsd_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtmsd(trajectory=traj, n=25)
    assert r.name
