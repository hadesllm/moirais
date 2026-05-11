"""Tests for morie.fn.nsccd — new-user active-comparator cohort."""
import numpy as np
import pytest
from morie.fn.nsccd import new_user_cohort


class TestNewUserCohort:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        rx_start = rng.uniform(0, 365, n)
        drug_group = np.array(["A"] * 100 + ["B"] * 100)
        outcome_time = rng.exponential(180, n)
        outcome_event = rng.binomial(1, 0.3, n)
        res = new_user_cohort(rx_start, drug_group, outcome_time, outcome_event)
        assert res.extra["hazard_ratio"] > 0
        assert res.extra["n_group_0"] > 0
        assert res.extra["n_group_1"] > 0

    def test_group_counts(self):
        rng = np.random.default_rng(42)
        n = 100
        rx_start = rng.uniform(0, 365, n)
        drug_group = np.array(["X"] * 60 + ["Y"] * 40)
        outcome_time = rng.exponential(200, n)
        outcome_event = rng.binomial(1, 0.2, n)
        res = new_user_cohort(rx_start, drug_group, outcome_time, outcome_event)
        total = res.extra["n_group_0"] + res.extra["n_group_1"]
        assert total == n

    def test_three_groups_raises(self):
        rng = np.random.default_rng(42)
        n = 90
        rx_start = rng.uniform(0, 365, n)
        drug_group = np.array(["A"] * 30 + ["B"] * 30 + ["C"] * 30)
        outcome_time = rng.exponential(180, n)
        outcome_event = rng.binomial(1, 0.3, n)
        with pytest.raises(ValueError):
            new_user_cohort(rx_start, drug_group, outcome_time, outcome_event)
