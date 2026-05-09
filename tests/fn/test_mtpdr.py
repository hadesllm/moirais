"""Test mtpdr."""
import numpy as np
import pytest
from moirais.fn.mtpdr import mtpdr


def test_mtpdr_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtpdr(trajectory=traj, n=25)
    assert r.value is not None


def test_mtpdr_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtpdr(trajectory=traj, n=25)
    assert r.name
