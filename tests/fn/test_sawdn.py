"""Test sawdn."""
import numpy as np
import pytest
from morie.fn.sawdn import sawdn


def test_sawdn_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawdn(values=vals, n=25)
    assert r.value is not None


def test_sawdn_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawdn(values=vals, n=25)
    assert r.name
