"""Test ppmkv."""
import numpy as np
import pytest
from morie.fn.ppmkv import ppmkv


def test_ppmkv_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmkv(points=pts, n=30)
    assert r.value is not None


def test_ppmkv_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmkv(points=pts, n=30)
    assert r.name
