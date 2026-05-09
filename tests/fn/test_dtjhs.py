"""Test dtjhs."""
import numpy as np
import pytest
from moirais.fn.dtjhs import dtjhs


def test_dtjhs_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtjhs(x=x, n=50)
    assert r.value is not None


def test_dtjhs_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtjhs(x=x, n=50)
    assert r.name
