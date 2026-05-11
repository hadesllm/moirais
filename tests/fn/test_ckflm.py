"""Test ckflm."""
import numpy as np
import pytest
from morie.fn.ckflm import ckflm


def test_ckflm_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = ckflm(x=x, y=y, values=v)
    assert r.value is not None


def test_ckflm_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = ckflm(x=x, y=y, values=v)
    assert r.name
