"""Test clcur."""
import numpy as np
import pytest
from morie.fn.clcur import clcur


def test_clcur_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcur(data=data, n=30, k=3)
    assert r.value is not None


def test_clcur_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcur(data=data, n=30, k=3)
    assert r.name
