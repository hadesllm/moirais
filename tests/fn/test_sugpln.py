"""Tests for morie.fn.sugpln — suggest analysis plan."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.sugpln import sugpln, suggest_analysis_plan
from morie.dataset import profile_dataset, DatasetProfile


def test_alias_is_same_function():
    """sugpln and suggest_analysis_plan are the same object."""
    assert sugpln is suggest_analysis_plan


@pytest.fixture()
def epi_profile():
    """Profile from a dataset with treatment, outcome, covariates."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "treatment": rng.choice([0, 1], n),
        "outcome": rng.standard_normal(n),
        "age": rng.integers(18, 80, n),
        "survey_wt": rng.uniform(0.5, 2.0, n),
        "gender": rng.choice(["M", "F"], n),
    })
    return profile_dataset(df)


def test_returns_list(epi_profile):
    """suggest_analysis_plan returns a list."""
    plan = sugpln(epi_profile)
    assert isinstance(plan, list)


def test_always_includes_descriptive(epi_profile):
    """Descriptive profile is always the first suggestion."""
    plan = sugpln(epi_profile)
    assert len(plan) > 0
    assert plan[0]["analysis"] == "descriptive_profile"


def test_includes_propensity_scores(epi_profile):
    """With binary treatment + covariates, propensity scores are suggested."""
    plan = sugpln(epi_profile)
    analyses = [s["analysis"] for s in plan]
    assert "propensity_scores" in analyses


def test_includes_ipw_ate(epi_profile):
    """With binary treatment + outcome, IPW ATE is suggested."""
    plan = sugpln(epi_profile)
    analyses = [s["analysis"] for s in plan]
    assert "ipw_ate" in analyses


def test_includes_survey_weights(epi_profile):
    """With survey weight column, survey-weighted estimation is suggested."""
    plan = sugpln(epi_profile)
    analyses = [s["analysis"] for s in plan]
    assert "survey_weighted_estimates" in analyses


def test_each_suggestion_has_keys(epi_profile):
    """Each suggestion has analysis, rationale, required_vars keys."""
    plan = sugpln(epi_profile)
    for s in plan:
        assert "analysis" in s
        assert "rationale" in s
        assert "required_vars" in s


def test_minimal_dataset_only_descriptive():
    """A dataset with no treatment/outcome only gets descriptive."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    profile = profile_dataset(df)
    plan = sugpln(profile)
    assert len(plan) == 1
    assert plan[0]["analysis"] == "descriptive_profile"
