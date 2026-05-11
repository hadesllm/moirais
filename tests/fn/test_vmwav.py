"""Test vmwav."""
import numpy as np
import pytest
from morie.fn.vmwav import vmwav


def test_vmwav_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmwav(x=x, y=y, values=v)
    assert r.value is not None


def test_vmwav_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vmwav(x=x, y=y, values=v)
    assert r.name
