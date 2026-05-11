"""Test dtbur."""
import numpy as np
import pytest
from morie.fn.dtbur import dtbur


def test_dtbur_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbur(x=x, n=50)
    assert r.value is not None


def test_dtbur_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbur(x=x, n=50)
    assert r.name
