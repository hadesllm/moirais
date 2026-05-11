"""Test vmlin."""
import numpy as np
import pytest
from morie.fn.vmlin import vmlin


def test_vmlin_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmlin(x=x, y=y, values=v)
    assert r.value is not None


def test_vmlin_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmlin(x=x, y=y, values=v)
    assert r.name
