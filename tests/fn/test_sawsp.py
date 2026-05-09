"""Test sawsp."""
import numpy as np
import pytest
from moirais.fn.sawsp import sawsp


def test_sawsp_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawsp(values=vals, n=25)
    assert r.value is not None


def test_sawsp_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawsp(values=vals, n=25)
    assert r.name
