"""Test coefficient_of_variation (coefv)."""
import numpy as np
import pytest

from moirais.fn.coefv import coefficient_of_variation, coefv
from moirais.fn._containers import DescriptiveResult


class TestCoefficientOfVariation:
    def test_basic(self):
        x = np.array([10.0, 12.0, 11.0, 13.0, 9.0])
        result = coefficient_of_variation(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "coefficient_of_variation"

    def test_positive_value(self):
        x = np.array([10.0, 12.0, 11.0, 13.0, 9.0])
        result = coefficient_of_variation(x)
        assert result.value > 0

    def test_zero_mean(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = coefficient_of_variation(x)
        assert result.value == float("inf")

    def test_alias(self):
        assert coefv is coefficient_of_variation
