"""Test sawkn."""
import numpy as np
import pytest
from morie.fn.sawkn import sawkn


def test_sawkn_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawkn(values=vals, n=25)
    assert r.value is not None


def test_sawkn_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawkn(values=vals, n=25)
    assert r.name
