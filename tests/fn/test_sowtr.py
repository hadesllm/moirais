"""Test sowtr."""
import numpy as np
import pytest
from moirais.fn.sowtr import sowtr


def test_sowtr_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = sowtr(data=data, depth=depth, n=20)
    assert r.value is not None


def test_sowtr_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = sowtr(data=data, depth=depth, n=20)
    assert r.name
