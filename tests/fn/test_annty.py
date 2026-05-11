"""Tests for annty (annuity present value)."""
from morie.fn.annty import annuity_value


def test_annuity_ordinary():
    r = annuity_value(rate=0.05, n_periods=10, payment=100)
    assert r.value > 0
    assert abs(r.value - 772.17) < 2.0


def test_annuity_due():
    r_ord = annuity_value(rate=0.05, n_periods=10, payment=100, due=False)
    r_due = annuity_value(rate=0.05, n_periods=10, payment=100, due=True)
    assert r_due.value > r_ord.value
