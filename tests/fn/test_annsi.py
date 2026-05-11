"""Test annsi."""
import numpy as np
import pytest
from morie.fn.annsi import annsi


def test_annsi_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = annsi(points=pts, n=40)
    assert r.value is not None


def test_annsi_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = annsi(points=pts, n=40)
    assert r.name
