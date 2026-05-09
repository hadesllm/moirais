"""Test dtctm."""
import numpy as np
import pytest
from moirais.fn.dtctm import dtctm


def test_dtctm_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtctm(x=x, n=50)
    assert r.value is not None


def test_dtctm_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtctm(x=x, n=50)
    assert r.name
