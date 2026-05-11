"""Test mteff."""
import numpy as np
import pytest
from morie.fn.mteff import mteff


def test_mteff_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mteff(trajectory=traj, n=25)
    assert r.value is not None


def test_mteff_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mteff(trajectory=traj, n=25)
    assert r.name
