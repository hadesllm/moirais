"""Test algorithm_convergence (alcon)."""
import numpy as np
import pytest

from moirais.fn.alcon import algorithm_convergence, alcon
from moirais.fn._containers import DescriptiveResult


class TestAlgorithmConvergence:
    def test_monotonic(self):
        costs = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = algorithm_convergence(costs)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "algorithm_convergence"
        assert result.extra["monotonic"] is True

    def test_non_monotonic(self):
        costs = [10.0, 8.0, 9.0, 7.0]
        result = algorithm_convergence(costs)
        assert result.extra["monotonic"] is False

    def test_total_reduction(self):
        costs = [10.0, 5.0, 1.0]
        result = algorithm_convergence(costs)
        assert result.extra["total_reduction"] == 9.0

    def test_too_few(self):
        with pytest.raises(ValueError):
            algorithm_convergence([5.0])

    def test_alias(self):
        assert alcon is algorithm_convergence
