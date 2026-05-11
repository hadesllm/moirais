"""Test markv."""
import numpy as np
import pytest
from morie.fn.markv import markv


def test_markv_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = markv(points=pts, n=40)
    assert r.value is not None


def test_markv_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = markv(points=pts, n=40)
    assert r.name
