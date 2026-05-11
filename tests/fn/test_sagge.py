"""Test sagge."""
import numpy as np
import pytest
from morie.fn.sagge import sagge


def test_sagge_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagge(values=vals, n=25)
    assert r.value is not None


def test_sagge_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagge(values=vals, n=25)
    assert r.name
