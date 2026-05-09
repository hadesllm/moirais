"""Test grdsmp."""
import numpy as np
import pytest
from moirais.fn.grdsmp import grdsmp


def test_grdsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = grdsmp(coords=coords, n=20)
    assert r.value is not None


def test_grdsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = grdsmp(coords=coords, n=20)
    assert r.name
