"""Test salml."""
import numpy as np
import pytest
from moirais.fn.salml import salml


def test_salml_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salml(values=vals, n=25)
    assert r.value is not None


def test_salml_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = salml(values=vals, n=25)
    assert r.name
