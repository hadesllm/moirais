"""Test samsp."""
import numpy as np
import pytest
from morie.fn.samsp import samsp


def test_samsp_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samsp(values=vals, n=25)
    assert r.value is not None


def test_samsp_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samsp(values=vals, n=25)
    assert r.name
