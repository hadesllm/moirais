"""Test mtdia."""
import numpy as np
import pytest
from moirais.fn.mtdia import mtdia


def test_mtdia_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtdia(trajectory=traj, n=25)
    assert r.value is not None


def test_mtdia_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtdia(trajectory=traj, n=25)
    assert r.name
