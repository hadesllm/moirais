"""Test sapmo."""
import numpy as np
import pytest
from moirais.fn.sapmo import sapmo


def test_sapmo_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sapmo(values=vals, n=25)
    assert r.value is not None


def test_sapmo_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sapmo(values=vals, n=25)
    assert r.name
