"""Test vmenv."""
import numpy as np
import pytest
from moirais.fn.vmenv import vmenv


def test_vmenv_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmenv(x=x, y=y, values=v)
    assert r.value is not None


def test_vmenv_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmenv(x=x, y=y, values=v)
    assert r.name
