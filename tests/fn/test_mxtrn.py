"""Tests for matrix_trace_norm."""
import numpy as np
import pytest
from morie.fn.mxtrn import matrix_trace_norm, mxtrn


def test_identity():
    r = matrix_trace_norm(np.eye(3))
    assert abs(r.extra["trace"] - 3.0) < 1e-10
    assert abs(r.extra["determinant"] - 1.0) < 1e-10
    assert r.extra["rank"] == 3


def test_alias():
    assert mxtrn is matrix_trace_norm


def test_not_square():
    with pytest.raises(ValueError):
        matrix_trace_norm(np.ones((2, 3)))
