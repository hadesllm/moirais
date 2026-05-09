"""Test salmo."""
import numpy as np
import pytest
from moirais.fn.salmo import salmo


def test_salmo_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salmo(values=vals, n=25)
    assert r.value is not None


def test_salmo_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salmo(values=vals, n=25)
    assert r.name
