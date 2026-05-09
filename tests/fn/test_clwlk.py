"""Test clwlk."""
import numpy as np
import pytest
from moirais.fn.clwlk import clwlk


def test_clwlk_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clwlk(data=data, n=30, k=3)
    assert r.value is not None


def test_clwlk_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clwlk(data=data, n=30, k=3)
    assert r.name
