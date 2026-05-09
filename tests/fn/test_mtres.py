"""Test mtres."""
import numpy as np
import pytest
from moirais.fn.mtres import mtres


def test_mtres_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtres(trajectory=traj, n=25)
    assert r.value is not None


def test_mtres_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtres(trajectory=traj, n=25)
    assert r.name
