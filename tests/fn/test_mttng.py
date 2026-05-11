"""Test mttng."""
import numpy as np
import pytest
from morie.fn.mttng import mttng


def test_mttng_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttng(trajectory=traj, n=25)
    assert r.value is not None


def test_mttng_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mttng(trajectory=traj, n=25)
    assert r.name
