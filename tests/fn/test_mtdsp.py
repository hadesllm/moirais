"""Test mtdsp."""
import numpy as np
import pytest
from morie.fn.mtdsp import mtdsp


def test_mtdsp_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtdsp(trajectory=traj, n=25)
    assert r.value is not None


def test_mtdsp_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtdsp(trajectory=traj, n=25)
    assert r.name
