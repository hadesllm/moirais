"""Test salge."""
import numpy as np
import pytest
from moirais.fn.salge import salge


def test_salge_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salge(values=vals, n=25)
    assert r.value is not None


def test_salge_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salge(values=vals, n=25)
    assert r.name
