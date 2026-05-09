"""Test mtfpt."""
import numpy as np
import pytest
from moirais.fn.mtfpt import mtfpt


def test_mtfpt_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtfpt(trajectory=traj, n=25)
    assert r.value is not None


def test_mtfpt_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtfpt(trajectory=traj, n=25)
    assert r.name
