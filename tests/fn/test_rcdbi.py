"""Tests for moirais.fn.rcdbi — recidivism burden."""

import pytest
import numpy as np
from moirais.fn.rcdbi import recidivism_burden
from moirais.fn._containers import ESRes


class TestRecidivismBurden:

    def test_returns_esres(self):
        counts = np.array([10, 5, 3])
        weights = np.array([1.0, 2.0, 3.0])
        result = recidivism_burden(counts, weights)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(10 * 1 + 5 * 2 + 3 * 3)

    def test_zero_counts(self):
        result = recidivism_burden(np.array([0, 0]), np.array([1.0, 2.0]))
        assert result.estimate == pytest.approx(0.0)
