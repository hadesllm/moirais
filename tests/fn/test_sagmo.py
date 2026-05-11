"""Test sagmo."""
import numpy as np
import pytest
from morie.fn.sagmo import sagmo


def test_sagmo_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagmo(values=vals, n=25)
    assert r.value is not None


def test_sagmo_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagmo(values=vals, n=25)
    assert r.name
