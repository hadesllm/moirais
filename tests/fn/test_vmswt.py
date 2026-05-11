"""Test vmswt."""
import numpy as np
import pytest
from morie.fn.vmswt import vmswt


def test_vmswt_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmswt(x=x, y=y, values=v)
    assert r.value is not None


def test_vmswt_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmswt(x=x, y=y, values=v)
    assert r.name
