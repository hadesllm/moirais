"""Test tbats."""
import numpy as np
import pytest
from morie.fn.tbats import tbats


def test_tbats_basic():
    """TBATS fit on simple series."""
    rng = np.random.default_rng(42)
    y = np.sin(np.arange(100) * 2 * np.pi / 12) + rng.standard_normal(100) * 0.1
    r = tbats(y, seasonal_periods=[12])
    assert r.fitted.shape == y.shape
    assert r.residuals.shape == y.shape


def test_tbats_box_cox():
    """TBATS with Box-Cox transformation."""
    rng = np.random.default_rng(42)
    y = np.exp(rng.standard_normal(80) * 0.5)
    r = tbats(y, use_box_cox=True)
    assert isinstance(r.lambda_bc, float)


def test_tbats_no_box_cox():
    """TBATS without Box-Cox."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(80)
    r = tbats(y, use_box_cox=False)
    assert r.lambda_bc == 1.0
