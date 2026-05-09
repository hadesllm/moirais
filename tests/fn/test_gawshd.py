"""Test gawshd."""
import numpy as np
import pytest
from moirais.fn.gawshd import gawshd


def test_gawshd_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = gawshd(x=x, y=y, values=v)
    assert r.value is not None


def test_gawshd_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 30)
    y = rng.uniform(0, 100, 30)
    v = rng.standard_normal(30)
    r = gawshd(x=x, y=y, values=v)
    assert r.name
