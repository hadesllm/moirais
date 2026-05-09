"""Test mtstp."""
import numpy as np
import pytest
from moirais.fn.mtstp import mtstp


def test_mtstp_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtstp(trajectory=traj, n=25)
    assert r.value is not None


def test_mtstp_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtstp(trajectory=traj, n=25)
    assert r.name
