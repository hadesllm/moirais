"""Test ppssp."""
import numpy as np
import pytest
from morie.fn.ppssp import ppssp


def test_ppssp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppssp(points=pts, n=30)
    assert r.value is not None


def test_ppssp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppssp(points=pts, n=30)
    assert r.name
