"""Test clagg."""
import numpy as np
import pytest
from moirais.fn.clagg import clagg


def test_clagg_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clagg(data=data, n=30, k=3)
    assert r.value is not None


def test_clagg_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clagg(data=data, n=30, k=3)
    assert r.name
