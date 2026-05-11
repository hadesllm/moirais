"""Test mtcnt."""
import numpy as np
import pytest
from morie.fn.mtcnt import mtcnt


def test_mtcnt_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcnt(trajectory=traj, n=25)
    assert r.value is not None


def test_mtcnt_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcnt(trajectory=traj, n=25)
    assert r.name
