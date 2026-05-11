"""Test vmjbs."""
import numpy as np
import pytest
from morie.fn.vmjbs import vmjbs


def test_vmjbs_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmjbs(x=x, y=y, values=v)
    assert r.value is not None


def test_vmjbs_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmjbs(x=x, y=y, values=v)
    assert r.name
