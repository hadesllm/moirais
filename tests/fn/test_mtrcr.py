"""Test mtrcr."""
import numpy as np
import pytest
from morie.fn.mtrcr import mtrcr


def test_mtrcr_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtrcr(trajectory=traj, n=25)
    assert r.value is not None


def test_mtrcr_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtrcr(trajectory=traj, n=25)
    assert r.name
