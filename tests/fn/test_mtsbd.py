"""Test mtsbd."""
import numpy as np
import pytest
from moirais.fn.mtsbd import mtsbd


def test_mtsbd_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtsbd(trajectory=traj, n=25)
    assert r.value is not None


def test_mtsbd_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtsbd(trajectory=traj, n=25)
    assert r.name
