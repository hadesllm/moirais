"""Test dtric."""
import numpy as np
import pytest
from moirais.fn.dtric import dtric


def test_dtric_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtric(x=x, n=50)
    assert r.value is not None


def test_dtric_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtric(x=x, n=50)
    assert r.name
