"""Test sawvr."""
import numpy as np
import pytest
from morie.fn.sawvr import sawvr


def test_sawvr_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawvr(values=vals, n=25)
    assert r.value is not None


def test_sawvr_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawvr(values=vals, n=25)
    assert r.name
