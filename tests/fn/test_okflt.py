"""Test okflt."""
import numpy as np
import pytest
from moirais.fn.okflt import okflt


def test_okflt_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = okflt(x=x, y=y, values=v)
    assert r.value is not None


def test_okflt_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = okflt(x=x, y=y, values=v)
    assert r.name
