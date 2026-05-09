"""Tests for moirais.fn.cvbnd — cross-validation risk bound."""

import numpy as np
import pytest

from moirais.fn.cvbnd import cvbnd


def test_basic_output():
    risks = np.array([0.5, 0.6, 0.55, 0.52, 0.58])
    result = cvbnd(risks, n=200)
    assert "cv_risk" in result
    assert "upper_bound" in result


def test_upper_bound_ge_risk():
    risks = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    result = cvbnd(risks, n=100)
    assert result["upper_bound"] >= result["cv_risk"]


def test_more_data_tighter():
    risks = np.array([0.5, 0.6, 0.55])
    r1 = cvbnd(risks, n=100)
    r2 = cvbnd(risks, n=10000)
    assert r2["bound_width"] < r1["bound_width"]


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        cvbnd(np.array([]), n=100)
