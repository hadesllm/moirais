"""Tests for irtlk -- IRT linking."""
import numpy as np
from morie.fn.irtlk import irt_linking
from morie.fn._containers import DescriptiveResult


class TestIrtLinking:
    def test_basic(self):
        a = {f"item_{j}": {"a": 1.0, "b": float(j)} for j in range(5)}
        b = {f"item_{j}": {"a": 1.0, "b": float(j) + 0.5} for j in range(5)}
        result = irt_linking(a, b)
        assert isinstance(result, DescriptiveResult)
        assert "A" in result.extra
        assert "B" in result.extra

    def test_linked_params_exist(self):
        a = {f"item_{j}": {"a": 1.0, "b": float(j)} for j in range(3)}
        b = {f"item_{j}": {"a": 1.0, "b": float(j) * 2} for j in range(3)}
        result = irt_linking(a, b)
        assert "linked_params" in result.extra
        assert len(result.extra["linked_params"]) == 3
