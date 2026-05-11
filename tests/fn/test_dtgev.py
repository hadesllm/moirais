"""Test dtgev."""
import numpy as np
import pytest
from morie.fn.dtgev import dtgev


def test_dtgev_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtgev(x=x, n=50)
    assert r.value is not None


def test_dtgev_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtgev(x=x, n=50)
    assert r.name
