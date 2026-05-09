"""Test dtjhm."""
import numpy as np
import pytest
from moirais.fn.dtjhm import dtjhm


def test_dtjhm_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtjhm(x=x, n=50)
    assert r.value is not None


def test_dtjhm_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtjhm(x=x, n=50)
    assert r.name
