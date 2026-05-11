"""Test mtnet."""
import numpy as np
import pytest
from morie.fn.mtnet import mtnet


def test_mtnet_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtnet(trajectory=traj, n=25)
    assert r.value is not None


def test_mtnet_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtnet(trajectory=traj, n=25)
    assert r.name
