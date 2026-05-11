"""Test dtwsh."""
import numpy as np
import pytest
from morie.fn.dtwsh import dtwsh


def test_dtwsh_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtwsh(x=x, n=50)
    assert r.value is not None


def test_dtwsh_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtwsh(x=x, n=50)
    assert r.name
