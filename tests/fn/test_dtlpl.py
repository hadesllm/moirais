"""Test dtlpl."""
import numpy as np
import pytest
from morie.fn.dtlpl import dtlpl


def test_dtlpl_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtlpl(x=x, n=50)
    assert r.value is not None


def test_dtlpl_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtlpl(x=x, n=50)
    assert r.name
