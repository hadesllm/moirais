"""Test difanl."""
import numpy as np
import pytest
from morie.fn.difanl import difanl


def test_difanl_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = difanl(x=x, y=y, values=v)
    assert r.value is not None


def test_difanl_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = difanl(x=x, y=y, values=v)
    assert r.name
