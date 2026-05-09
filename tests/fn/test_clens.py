"""Test clens."""
import numpy as np
import pytest
from moirais.fn.clens import clens


def test_clens_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clens(data=data, n=30, k=3)
    assert r.value is not None


def test_clens_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clens(data=data, n=30, k=3)
    assert r.name
