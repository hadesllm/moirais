"""Test sagjw."""
import numpy as np
import pytest
from morie.fn.sagjw import sagjw


def test_sagjw_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagjw(values=vals, n=25)
    assert r.value is not None


def test_sagjw_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagjw(values=vals, n=25)
    assert r.name
