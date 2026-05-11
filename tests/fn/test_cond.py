"""Tests for morie.fn.cond — condition number."""
import numpy as np
import pytest
from morie.fn.cond import condition_number


class TestConditionNumber:
    def test_identity(self):
        res = condition_number(np.eye(5))
        assert res.estimate == pytest.approx(1.0, abs=1e-10)

    def test_ill_conditioned(self):
        A = np.array([[1, 1], [1, 1.0001]])
        res = condition_number(A)
        assert res.estimate > 1000
