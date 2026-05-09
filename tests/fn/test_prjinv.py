"""Test prjinv."""
import numpy as np
import pytest
from moirais.fn.prjinv import prjinv


def test_prjinv_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = prjinv(coords=coords, n=20)
    assert r.value is not None


def test_prjinv_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = prjinv(coords=coords, n=20)
    assert r.name
