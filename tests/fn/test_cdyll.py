"""Tests for morie.fn.cdyll -- YLL."""

import pytest
import numpy as np
from morie.fn.cdyll import years_life_lost


class TestYLL:
    def test_scalar(self):
        res = years_life_lost(deaths=10, life_expectancy_remaining=30.0)
        assert res.estimate == pytest.approx(300.0)

    def test_array(self):
        res = years_life_lost(
            deaths=np.array([5, 10]),
            life_expectancy_remaining=np.array([40.0, 20.0]),
        )
        assert res.estimate == pytest.approx(400.0)
