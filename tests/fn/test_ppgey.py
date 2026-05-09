"""Test ppgey."""
import numpy as np
import pytest
from moirais.fn.ppgey import ppgey


def test_ppgey_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppgey(points=pts, n=30)
    assert r.value is not None


def test_ppgey_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppgey(points=pts, n=30)
    assert r.name
