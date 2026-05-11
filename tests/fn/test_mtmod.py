"""Test mtmod."""
import numpy as np
import pytest
from morie.fn.mtmod import mtmod


def test_mtmod_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtmod(trajectory=traj, n=25)
    assert r.value is not None


def test_mtmod_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtmod(trajectory=traj, n=25)
    assert r.name
