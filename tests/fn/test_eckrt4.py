"""Test eckrt4."""
import numpy as np
import pytest
from moirais.fn.eckrt4 import eckrt4


def test_eckrt4_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = eckrt4(coords=coords, n=20)
    assert r.value is not None


def test_eckrt4_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = eckrt4(coords=coords, n=20)
    assert r.name
