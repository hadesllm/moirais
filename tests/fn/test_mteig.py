"""Test mteig."""
import numpy as np
import pytest
from moirais.fn.mteig import mteig


def test_mteig_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mteig(trajectory=traj, n=25)
    assert r.value is not None


def test_mteig_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mteig(trajectory=traj, n=25)
    assert r.name
