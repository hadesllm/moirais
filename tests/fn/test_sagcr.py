"""Test sagcr."""
import numpy as np
import pytest
from morie.fn.sagcr import sagcr


def test_sagcr_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagcr(values=vals, n=25)
    assert r.value is not None


def test_sagcr_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagcr(values=vals, n=25)
    assert r.name
