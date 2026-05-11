"""Test sawco."""
import numpy as np
import pytest
from morie.fn.sawco import sawco


def test_sawco_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawco(values=vals, n=25)
    assert r.value is not None


def test_sawco_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawco(values=vals, n=25)
    assert r.name
