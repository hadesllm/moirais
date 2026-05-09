"""Test slcsp."""
import numpy as np
import pytest
from moirais.fn.slcsp import slcsp


def test_slcsp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = slcsp(points=pts, n=40)
    assert r.value is not None


def test_slcsp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = slcsp(points=pts, n=40)
    assert r.name
