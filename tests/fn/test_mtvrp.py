"""Test mtvrp."""
import numpy as np
import pytest
from morie.fn.mtvrp import mtvrp


def test_mtvrp_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtvrp(trajectory=traj, n=25)
    assert r.value is not None


def test_mtvrp_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtvrp(trajectory=traj, n=25)
    assert r.name
