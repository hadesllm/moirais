"""Test sinusd."""
import numpy as np
import pytest
from morie.fn.sinusd import sinusd


def test_sinusd_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = sinusd(coords=coords, n=20)
    assert r.value is not None


def test_sinusd_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = sinusd(coords=coords, n=20)
    assert r.name
