"""Test mtcls."""
import numpy as np
import pytest
from moirais.fn.mtcls import mtcls


def test_mtcls_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcls(trajectory=traj, n=25)
    assert r.value is not None


def test_mtcls_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtcls(trajectory=traj, n=25)
    assert r.name
