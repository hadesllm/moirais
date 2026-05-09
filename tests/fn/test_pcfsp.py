"""Test pcfsp."""
import numpy as np
import pytest
from moirais.fn.pcfsp import pcfsp


def test_pcfsp_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = pcfsp(points=pts, n=40)
    assert r.value is not None


def test_pcfsp_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = pcfsp(points=pts, n=40)
    assert r.name
