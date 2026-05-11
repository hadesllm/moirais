"""Test dtwrp."""
import numpy as np
import pytest
from morie.fn.dtwrp import dtwrp


def test_dtwrp_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtwrp(x=x, n=50)
    assert r.value is not None


def test_dtwrp_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtwrp(x=x, n=50)
    assert r.name
