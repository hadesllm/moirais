"""Test sawad."""
import numpy as np
import pytest
from morie.fn.sawad import sawad


def test_sawad_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawad(values=vals, n=25)
    assert r.value is not None


def test_sawad_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawad(values=vals, n=25)
    assert r.name
