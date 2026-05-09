"""Test dtdag."""
import numpy as np
import pytest
from moirais.fn.dtdag import dtdag


def test_dtdag_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtdag(x=x, n=50)
    assert r.value is not None


def test_dtdag_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtdag(x=x, n=50)
    assert r.name
