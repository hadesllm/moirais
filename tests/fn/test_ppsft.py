"""Test ppsft."""
import numpy as np
import pytest
from morie.fn.ppsft import ppsft


def test_ppsft_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppsft(points=pts, n=30)
    assert r.value is not None


def test_ppsft_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppsft(points=pts, n=30)
    assert r.name
