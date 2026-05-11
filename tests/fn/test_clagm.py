"""Test clagm."""
import numpy as np
import pytest
from morie.fn.clagm import clagm


def test_clagm_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clagm(data=data, n=30, k=3)
    assert r.value is not None


def test_clagm_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clagm(data=data, n=30, k=3)
    assert r.name
