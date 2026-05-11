"""Test krgre."""
import numpy as np
import pytest
from morie.fn.krgre import krgre


def test_krgre_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = krgre(x=x, y=y, values=v)
    assert r.value is not None


def test_krgre_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = krgre(x=x, y=y, values=v)
    assert r.name
