"""Test goodel."""
import numpy as np
import pytest
from moirais.fn.goodel import goodel


def test_goodel_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = goodel(coords=coords, n=20)
    assert r.value is not None


def test_goodel_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = goodel(coords=coords, n=20)
    assert r.name
