"""Test winklt."""
import numpy as np
import pytest
from moirais.fn.winklt import winklt


def test_winklt_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = winklt(coords=coords, n=20)
    assert r.value is not None


def test_winklt_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = winklt(coords=coords, n=20)
    assert r.name
