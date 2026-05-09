"""Test vmeff."""
import numpy as np
import pytest
from moirais.fn.vmeff import vmeff


def test_vmeff_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmeff(x=x, y=y, values=v)
    assert r.value is not None


def test_vmeff_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmeff(x=x, y=y, values=v)
    assert r.name
