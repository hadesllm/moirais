"""Test mtssm."""
import numpy as np
import pytest
from moirais.fn.mtssm import mtssm


def test_mtssm_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtssm(trajectory=traj, n=25)
    assert r.value is not None


def test_mtssm_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtssm(trajectory=traj, n=25)
    assert r.name
