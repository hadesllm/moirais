"""Test mttsp2."""
import numpy as np
import pytest
from morie.fn.mttsp2 import mttsp2


def test_mttsp2_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttsp2(trajectory=traj, n=25)
    assert r.value is not None


def test_mttsp2_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttsp2(trajectory=traj, n=25)
    assert r.name
