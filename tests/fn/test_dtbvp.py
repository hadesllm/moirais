"""Test dtbvp."""
import numpy as np
import pytest
from morie.fn.dtbvp import dtbvp


def test_dtbvp_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbvp(x=x, n=50)
    assert r.value is not None


def test_dtbvp_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbvp(x=x, n=50)
    assert r.name
