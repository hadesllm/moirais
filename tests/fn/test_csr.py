"""Test csr."""
import numpy as np
import pytest
from morie.fn.csr import csr


def test_csr_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = csr(points=pts, n=40)
    assert isinstance(r.value, float)
    assert r.value > 0, "Mean nearest-neighbor distance must be positive"
    assert r.value < 100, "Mean NN distance implausibly large for 100x100 window"


def test_csr_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = csr(points=pts, n=40)
    assert r.name
