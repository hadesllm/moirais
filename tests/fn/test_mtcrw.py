"""Test mtcrw."""
import numpy as np
import pytest
from moirais.fn.mtcrw import mtcrw


def test_mtcrw_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcrw(trajectory=traj, n=25)
    assert r.value is not None


def test_mtcrw_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcrw(trajectory=traj, n=25)
    assert r.name
