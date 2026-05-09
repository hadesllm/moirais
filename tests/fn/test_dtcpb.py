"""Test dtcpb."""
import numpy as np
import pytest
from moirais.fn.dtcpb import dtcpb


def test_dtcpb_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpb(x=x, n=50)
    assert r.value is not None


def test_dtcpb_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpb(x=x, n=50)
    assert r.name
