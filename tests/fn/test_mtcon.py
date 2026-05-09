"""Test mtcon."""
import numpy as np
import pytest
from moirais.fn.mtcon import mtcon


def test_mtcon_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcon(trajectory=traj, n=25)
    assert r.value is not None


def test_mtcon_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcon(trajectory=traj, n=25)
    assert r.name
