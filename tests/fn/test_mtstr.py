"""Test mtstr."""
import numpy as np
import pytest
from morie.fn.mtstr import mtstr


def test_mtstr_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtstr(trajectory=traj, n=25)
    assert r.value is not None


def test_mtstr_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtstr(trajectory=traj, n=25)
    assert r.name
