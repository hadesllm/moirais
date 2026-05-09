"""Test hmcsp."""
import numpy as np
import pytest
from moirais.fn.hmcsp import hmcsp


def test_hmcsp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = hmcsp(points=pts, n=40)
    assert r.value is not None


def test_hmcsp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = hmcsp(points=pts, n=40)
    assert r.name
