"""Test dtprt."""
import numpy as np
import pytest
from morie.fn.dtprt import dtprt


def test_dtprt_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtprt(x=x, n=50)
    assert r.value is not None


def test_dtprt_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtprt(x=x, n=50)
    assert r.name
