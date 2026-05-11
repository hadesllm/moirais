"""Test mtdns."""
import numpy as np
import pytest
from morie.fn.mtdns import mtdns


def test_mtdns_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtdns(trajectory=traj, n=25)
    assert r.value is not None


def test_mtdns_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtdns(trajectory=traj, n=25)
    assert r.name
