"""Test dtwrn."""
import numpy as np
import pytest
from morie.fn.dtwrn import dtwrn


def test_dtwrn_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtwrn(x=x, n=50)
    assert r.value is not None


def test_dtwrn_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtwrn(x=x, n=50)
    assert r.name
