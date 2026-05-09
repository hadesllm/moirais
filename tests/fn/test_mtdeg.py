"""Test mtdeg."""
import numpy as np
import pytest
from moirais.fn.mtdeg import mtdeg


def test_mtdeg_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtdeg(trajectory=traj, n=25)
    assert r.value is not None


def test_mtdeg_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtdeg(trajectory=traj, n=25)
    assert r.name
