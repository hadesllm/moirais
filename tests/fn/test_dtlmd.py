"""Test dtlmd."""
import numpy as np
import pytest
from morie.fn.dtlmd import dtlmd


def test_dtlmd_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtlmd(x=x, n=50)
    assert r.value is not None


def test_dtlmd_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtlmd(x=x, n=50)
    assert r.name
