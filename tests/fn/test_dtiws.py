"""Test dtiws."""
import numpy as np
import pytest
from moirais.fn.dtiws import dtiws


def test_dtiws_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtiws(x=x, n=50)
    assert r.value is not None


def test_dtiws_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtiws(x=x, n=50)
    assert r.name
