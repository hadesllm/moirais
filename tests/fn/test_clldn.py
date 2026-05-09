"""Test clldn."""
import numpy as np
import pytest
from moirais.fn.clldn import clldn


def test_clldn_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clldn(data=data, n=30, k=3)
    assert r.value is not None


def test_clldn_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clldn(data=data, n=30, k=3)
    assert r.name
