"""Test dtrlg."""
import numpy as np
import pytest
from morie.fn.dtrlg import dtrlg


def test_dtrlg_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtrlg(x=x, n=50)
    assert r.value is not None


def test_dtrlg_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtrlg(x=x, n=50)
    assert r.name
