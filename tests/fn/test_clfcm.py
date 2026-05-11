"""Test clfcm."""
import numpy as np
import pytest
from morie.fn.clfcm import clfcm


def test_clfcm_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clfcm(data=data, n=30, k=3)
    assert r.value is not None


def test_clfcm_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clfcm(data=data, n=30, k=3)
    assert r.name
