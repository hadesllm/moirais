"""Test mtinf."""
import numpy as np
import pytest
from morie.fn.mtinf import mtinf


def test_mtinf_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtinf(trajectory=traj, n=25)
    assert r.value is not None


def test_mtinf_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtinf(trajectory=traj, n=25)
    assert r.name
