"""Test dtcpt."""
import numpy as np
import pytest
from moirais.fn.dtcpt import dtcpt


def test_dtcpt_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpt(x=x, n=50)
    assert r.value is not None


def test_dtcpt_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpt(x=x, n=50)
    assert r.name
