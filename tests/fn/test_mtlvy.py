"""Test mtlvy."""
import numpy as np
import pytest
from moirais.fn.mtlvy import mtlvy


def test_mtlvy_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtlvy(trajectory=traj, n=25)
    assert isinstance(r.value, float)
    assert r.value > 0, "Mean step length must be positive"
    assert np.isfinite(r.value), "Mean step length must be finite"


def test_mtlvy_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtlvy(trajectory=traj, n=25)
    assert r.name
