"""Test mtbtw."""
import numpy as np
import pytest
from moirais.fn.mtbtw import mtbtw


def test_mtbtw_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtbtw(trajectory=traj, n=25)
    assert r.value is not None


def test_mtbtw_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtbtw(trajectory=traj, n=25)
    assert r.name
