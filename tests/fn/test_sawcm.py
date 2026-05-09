"""Test sawcm."""
import numpy as np
import pytest
from moirais.fn.sawcm import sawcm


def test_sawcm_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawcm(values=vals, n=25)
    assert r.value is not None


def test_sawcm_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawcm(values=vals, n=25)
    assert r.name
