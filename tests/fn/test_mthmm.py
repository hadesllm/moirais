"""Test mthmm."""
import numpy as np
import pytest
from morie.fn.mthmm import mthmm


def test_mthmm_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mthmm(trajectory=traj, n=25)
    assert r.value is not None


def test_mthmm_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mthmm(trajectory=traj, n=25)
    assert r.name
