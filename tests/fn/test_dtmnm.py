"""Test dtmnm."""
import numpy as np
import pytest
from moirais.fn.dtmnm import dtmnm


def test_dtmnm_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmnm(x=x, n=50)
    assert r.value is not None


def test_dtmnm_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmnm(x=x, n=50)
    assert r.name
