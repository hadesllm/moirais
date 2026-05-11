"""Test mtcps."""
import numpy as np
import pytest
from morie.fn.mtcps import mtcps


def test_mtcps_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcps(trajectory=traj, n=25)
    assert r.value is not None


def test_mtcps_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcps(trajectory=traj, n=25)
    assert r.name
