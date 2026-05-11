"""Test clgmx."""
import numpy as np
import pytest
from morie.fn.clgmx import clgmx


def test_clgmx_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clgmx(data=data, n=30, k=3)
    assert r.value is not None


def test_clgmx_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clgmx(data=data, n=30, k=3)
    assert r.name
