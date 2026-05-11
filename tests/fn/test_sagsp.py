"""Test sagsp."""
import numpy as np
import pytest
from morie.fn.sagsp import sagsp


def test_sagsp_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagsp(values=vals, n=25)
    assert r.value is not None


def test_sagsp_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagsp(values=vals, n=25)
    assert r.name
