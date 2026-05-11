"""Test mtcor."""
import numpy as np
import pytest
from morie.fn.mtcor import mtcor


def test_mtcor_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcor(trajectory=traj, n=25)
    assert r.value is not None


def test_mtcor_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcor(trajectory=traj, n=25)
    assert r.name
