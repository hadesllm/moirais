"""Test clkmn."""
import numpy as np
import pytest
from moirais.fn.clkmn import clkmn


def test_clkmn_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clkmn(data=data, n=30, k=3)
    assert r.value is not None


def test_clkmn_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clkmn(data=data, n=30, k=3)
    assert r.name
