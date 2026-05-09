"""Test pppnl."""
import numpy as np
import pytest
from moirais.fn.pppnl import pppnl


def test_pppnl_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pppnl(points=pts, n=30)
    assert r.value is not None


def test_pppnl_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = pppnl(points=pts, n=30)
    assert r.name
