"""Tests for morie.fn.cdprv -- chronic disease prevalence."""

import pytest
import numpy as np
from morie.fn.cdprv import chronic_disease_prevalence


class TestChronicDiseasePrevalence:
    def test_crude(self):
        res = chronic_disease_prevalence(n_cases=100, population=1000)
        assert res.estimate == pytest.approx(0.1)

    def test_age_adjusted(self):
        res = chronic_disease_prevalence(
            n_cases=np.array([10, 30]),
            population=np.array([200, 300]),
            age_groups=np.array(["young", "old"]),
            std_pop=np.array([0.6, 0.4]),
        )
        assert res.measure == "age_adjusted_prevalence"
        assert 0 < res.estimate < 1
