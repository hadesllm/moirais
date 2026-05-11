"""Test sawgb."""
import numpy as np
import pytest
from morie.fn.sawgb import sawgb


def test_sawgb_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawgb(values=vals, n=25)
    assert r.value is not None


def test_sawgb_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawgb(values=vals, n=25)
    assert r.name
