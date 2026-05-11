"""Test dtcir."""
import numpy as np
import pytest
from morie.fn.dtcir import dtcir


def test_dtcir_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcir(x=x, n=50)
    assert r.value is not None


def test_dtcir_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcir(x=x, n=50)
    assert r.name
