"""Test saqmo."""
import numpy as np
import pytest
from moirais.fn.saqmo import saqmo


def test_saqmo_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saqmo(values=vals, n=25)
    assert r.value is not None


def test_saqmo_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saqmo(values=vals, n=25)
    assert r.name
