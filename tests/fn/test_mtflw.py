"""Test mtflw."""
import numpy as np
import pytest
from morie.fn.mtflw import mtflw


def test_mtflw_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtflw(trajectory=traj, n=25)
    assert r.value is not None


def test_mtflw_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtflw(trajectory=traj, n=25)
    assert r.name
