"""Test mtsin."""
import numpy as np
import pytest
from moirais.fn.mtsin import mtsin


def test_mtsin_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtsin(trajectory=traj, n=25)
    assert r.value is not None


def test_mtsin_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtsin(trajectory=traj, n=25)
    assert r.name
