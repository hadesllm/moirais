"""Test vorone."""
import numpy as np
import pytest
from moirais.fn.vorone import vorone


def test_vorone_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vorone(x=x, y=y, values=v)
    assert r.value is not None


def test_vorone_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = vorone(x=x, y=y, values=v)
    assert r.name
