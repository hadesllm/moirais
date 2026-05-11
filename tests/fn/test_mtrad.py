"""Test mtrad."""
import numpy as np
import pytest
from morie.fn.mtrad import mtrad


def test_mtrad_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtrad(trajectory=traj, n=25)
    assert r.value is not None


def test_mtrad_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtrad(trajectory=traj, n=25)
    assert r.name
