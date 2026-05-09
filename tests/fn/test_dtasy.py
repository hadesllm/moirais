"""Test dtasy."""
import numpy as np
import pytest
from moirais.fn.dtasy import dtasy


def test_dtasy_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtasy(x=x, n=50)
    assert r.value is not None


def test_dtasy_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtasy(x=x, n=50)
    assert r.name
