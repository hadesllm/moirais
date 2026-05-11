"""Test dtnkc."""
import numpy as np
import pytest
from morie.fn.dtnkc import dtnkc


def test_dtnkc_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtnkc(x=x, n=50)
    assert r.value is not None


def test_dtnkc_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtnkc(x=x, n=50)
    assert r.name
