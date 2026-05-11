"""Test saldf."""
import numpy as np
import pytest
from morie.fn.saldf import saldf


def test_saldf_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saldf(values=vals, n=25)
    assert r.value is not None


def test_saldf_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saldf(values=vals, n=25)
    assert r.name
