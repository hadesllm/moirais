"""Test mtred."""
import numpy as np
import pytest
from moirais.fn.mtred import mtred


def test_mtred_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtred(trajectory=traj, n=25)
    assert r.value is not None


def test_mtred_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtred(trajectory=traj, n=25)
    assert r.name
