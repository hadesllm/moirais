"""Test sagjo."""
import numpy as np
import pytest
from moirais.fn.sagjo import sagjo


def test_sagjo_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagjo(values=vals, n=25)
    assert r.value is not None


def test_sagjo_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagjo(values=vals, n=25)
    assert r.name
