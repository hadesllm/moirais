"""Tests for moirais.fn.doseq -- Dose-response analysis."""

import pytest
from moirais.fn.doseq import dose_response


class TestDoseResponse:
    def test_basic(self):
        doses = [1, 2, 3, 4, 5]
        responses = [1, 3, 7, 15, 19]
        totals = [20, 20, 20, 20, 20]
        res = dose_response(doses, responses, totals)
        assert res.measure == "dose_response"
        assert res.extra["converged"]

    def test_ld50(self):
        doses = [1, 2, 3, 4, 5]
        responses = [2, 5, 10, 15, 18]
        totals = [20, 20, 20, 20, 20]
        res = dose_response(doses, responses, totals)
        assert 1 < res.estimate < 5

    def test_too_few(self):
        with pytest.raises(ValueError):
            dose_response([1], [5], [10])
