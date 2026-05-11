"""Test dtcar."""
import numpy as np
import pytest
from morie.fn.dtcar import dtcar


def test_dtcar_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcar(x=x, n=50)
    assert r.value is not None


def test_dtcar_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcar(x=x, n=50)
    assert r.name
