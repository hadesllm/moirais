"""Test ppmrk."""
import numpy as np
import pytest
from moirais.fn.ppmrk import ppmrk


def test_ppmrk_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmrk(points=pts, n=30)
    assert r.value is not None


def test_ppmrk_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmrk(points=pts, n=30)
    assert r.name
