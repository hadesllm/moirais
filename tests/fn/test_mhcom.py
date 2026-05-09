"""Tests for moirais.fn.mhcom -- comorbidity index."""

import pytest
import numpy as np
from moirais.fn.mhcom import comorbidity_index


class TestComorbidityIndex:
    def test_single_person(self):
        res = comorbidity_index([1, 0, 1, 0, 1])
        assert res.value == 3.0

    def test_population(self):
        data = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        res = comorbidity_index(data)
        assert res.value == pytest.approx(2.0)
        assert res.extra["pct_2plus"] == pytest.approx(2 / 3 * 100)
