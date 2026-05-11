"""Test mtbrw."""
import numpy as np
import pytest
from morie.fn.mtbrw import mtbrw


def test_mtbrw_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtbrw(trajectory=traj, n=25)
    assert r.value is not None


def test_mtbrw_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtbrw(trajectory=traj, n=25)
    assert r.name
