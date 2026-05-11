"""Test sawkr."""
import numpy as np
import pytest
from morie.fn.sawkr import sawkr


def test_sawkr_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawkr(values=vals, n=25)
    assert r.value is not None


def test_sawkr_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawkr(values=vals, n=25)
    assert r.name
