"""Test mttwr."""
import numpy as np
import pytest
from morie.fn.mttwr import mttwr


def test_mttwr_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttwr(trajectory=traj, n=25)
    assert r.value is not None


def test_mttwr_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttwr(trajectory=traj, n=25)
    assert r.name
