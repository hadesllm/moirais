"""Test mttrs."""
import numpy as np
import pytest
from moirais.fn.mttrs import mttrs


def test_mttrs_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttrs(trajectory=traj, n=25)
    assert r.value is not None


def test_mttrs_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttrs(trajectory=traj, n=25)
    assert r.name
