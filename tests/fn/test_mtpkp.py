"""Test mtpkp."""
import numpy as np
import pytest
from moirais.fn.mtpkp import mtpkp


def test_mtpkp_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtpkp(trajectory=traj, n=25)
    assert r.value is not None


def test_mtpkp_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtpkp(trajectory=traj, n=25)
    assert r.name
