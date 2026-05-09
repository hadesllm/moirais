"""Test mtarf."""
import numpy as np
import pytest
from moirais.fn.mtarf import mtarf


def test_mtarf_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtarf(trajectory=traj, n=25)
    assert r.value is not None


def test_mtarf_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtarf(trajectory=traj, n=25)
    assert r.name
