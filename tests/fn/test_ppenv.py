"""Test ppenv."""
import numpy as np
import pytest
from moirais.fn.ppenv import ppenv


def test_ppenv_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppenv(points=pts, n=30)
    assert r.value is not None


def test_ppenv_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppenv(points=pts, n=30)
    assert r.name
