"""Test dtfrc."""
import numpy as np
import pytest
from moirais.fn.dtfrc import dtfrc


def test_dtfrc_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtfrc(x=x, n=50)
    assert r.value is not None


def test_dtfrc_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtfrc(x=x, n=50)
    assert r.name
