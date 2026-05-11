"""Test mtrob."""
import numpy as np
import pytest
from morie.fn.mtrob import mtrob


def test_mtrob_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtrob(trajectory=traj, n=25)
    assert r.value is not None


def test_mtrob_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtrob(trajectory=traj, n=25)
    assert r.name
