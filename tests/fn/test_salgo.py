"""Test salgo."""
import numpy as np
import pytest
from morie.fn.salgo import salgo


def test_salgo_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salgo(values=vals, n=25)
    assert r.value is not None


def test_salgo_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salgo(values=vals, n=25)
    assert r.name
