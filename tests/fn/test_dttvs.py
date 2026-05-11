"""Test dttvs."""
import numpy as np
import pytest
from morie.fn.dttvs import dttvs


def test_dttvs_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dttvs(x=x, n=50)
    assert r.value is not None


def test_dttvs_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dttvs(x=x, n=50)
    assert r.name
