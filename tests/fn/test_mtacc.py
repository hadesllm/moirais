"""Test mtacc."""
import numpy as np
import pytest
from morie.fn.mtacc import mtacc


def test_mtacc_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtacc(trajectory=traj, n=25)
    assert r.value is not None


def test_mtacc_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtacc(trajectory=traj, n=25)
    assert r.name
