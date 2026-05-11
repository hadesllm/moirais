"""Test clcha."""
import numpy as np
import pytest
from morie.fn.clcha import clcha


def test_clcha_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcha(data=data, n=30, k=3)
    assert r.value is not None


def test_clcha_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcha(data=data, n=30, k=3)
    assert r.name
