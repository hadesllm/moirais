"""Test dtwbl2."""
import numpy as np
import pytest
from morie.fn.dtwbl2 import dtwbl2


def test_dtwbl2_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtwbl2(x=x, n=50)
    assert r.value is not None


def test_dtwbl2_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtwbl2(x=x, n=50)
    assert r.name
