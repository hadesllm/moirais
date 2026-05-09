"""Tests for moirais.fn.nobll -- bill outcome positions."""
import numpy as np
from moirais.fn.nobll import nominate_bill_params, nobll


def test_alias():
    assert nobll is nominate_bill_params


def test_smoke():
    nv = np.array([0.5, -0.3, 0.8])
    mid = np.array([0.0, 0.2, -0.1])
    r = nominate_bill_params(nv, mid)
    assert r.name == "nominate_bill_params"
    assert "yea_positions" in r.extra
    assert "nay_positions" in r.extra
    assert r.extra["n_bills"] == 3
