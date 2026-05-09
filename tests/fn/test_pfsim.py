"""Test pfsim."""
import numpy as np
import pytest
from moirais.fn.pfsim import pfsim


def test_pfsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = pfsim(points=pts, n=40)
    assert r.value is not None


def test_pfsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = pfsim(points=pts, n=40)
    assert r.name
