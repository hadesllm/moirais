"""Test mtent."""
import numpy as np
import pytest
from moirais.fn.mtent import mtent


def test_mtent_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtent(trajectory=traj, n=25)
    assert r.value is not None


def test_mtent_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtent(trajectory=traj, n=25)
    assert r.name
