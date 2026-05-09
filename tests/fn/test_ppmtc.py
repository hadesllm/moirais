"""Test ppmtc."""
import numpy as np
import pytest
from moirais.fn.ppmtc import ppmtc


def test_ppmtc_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmtc(points=pts, n=30)
    assert r.value is not None


def test_ppmtc_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmtc(points=pts, n=30)
    assert r.name
