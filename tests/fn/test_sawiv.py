"""Test sawiv."""
import numpy as np
import pytest
from morie.fn.sawiv import sawiv


def test_sawiv_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawiv(values=vals, n=25)
    assert r.value is not None


def test_sawiv_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawiv(values=vals, n=25)
    assert r.name
