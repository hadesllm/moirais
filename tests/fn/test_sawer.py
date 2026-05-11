"""Test sawer."""
import numpy as np
import pytest
from morie.fn.sawer import sawer


def test_sawer_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawer(values=vals, n=25)
    assert r.value is not None


def test_sawer_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawer(values=vals, n=25)
    assert r.name
