"""Test dtord."""
import numpy as np
import pytest
from morie.fn.dtord import dtord


def test_dtord_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtord(x=x, n=50)
    assert r.value is not None


def test_dtord_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtord(x=x, n=50)
    assert r.name
