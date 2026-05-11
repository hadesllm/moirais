"""Test sawrl."""
import numpy as np
import pytest
from morie.fn.sawrl import sawrl


def test_sawrl_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawrl(values=vals, n=25)
    assert r.value is not None


def test_sawrl_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawrl(values=vals, n=25)
    assert r.name
