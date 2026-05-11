"""Test vmrob."""
import numpy as np
import pytest
from morie.fn.vmrob import vmrob


def test_vmrob_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmrob(x=x, y=y, values=v)
    assert r.value is not None


def test_vmrob_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmrob(x=x, y=y, values=v)
    assert r.name
