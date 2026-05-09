"""Test vmind."""
import numpy as np
import pytest
from moirais.fn.vmind import vmind


def test_vmind_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmind(x=x, y=y, values=v)
    assert r.value is not None


def test_vmind_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmind(x=x, y=y, values=v)
    assert r.name
