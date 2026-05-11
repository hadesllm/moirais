"""Test mttrc."""
import numpy as np
import pytest
from morie.fn.mttrc import mttrc


def test_mttrc_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttrc(trajectory=traj, n=25)
    assert r.value is not None


def test_mttrc_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttrc(trajectory=traj, n=25)
    assert r.name
