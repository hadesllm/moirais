"""Test dtbng."""
import numpy as np
import pytest
from moirais.fn.dtbng import dtbng


def test_dtbng_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbng(x=x, n=50)
    assert r.value is not None


def test_dtbng_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbng(x=x, n=50)
    assert r.name
