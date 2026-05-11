"""Test balsmp."""
import numpy as np
import pytest
from morie.fn.balsmp import balsmp


def test_balsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = balsmp(coords=coords, n=20)
    assert r.value is not None


def test_balsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = balsmp(coords=coords, n=20)
    assert r.name
