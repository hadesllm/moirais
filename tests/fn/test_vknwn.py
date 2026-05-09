"""Tests for moirais.fn.vknwn — Known-groups validity."""

import numpy as np
import pytest
from moirais.fn.vknwn import validity_known_groups


class TestValidityKnownGroups:

    def test_two_groups_ttest(self, rng):
        scores = np.concatenate([rng.normal(10, 1, 50), rng.normal(15, 1, 50)])
        groups = np.array(["A"] * 50 + ["B"] * 50)
        result = validity_known_groups(scores, groups)
        assert result["test"] == "t-test"
        assert result["p_value"] < 0.05

    def test_three_groups_anova(self, rng):
        scores = np.concatenate([rng.normal(10, 1, 50),
                                 rng.normal(12, 1, 50),
                                 rng.normal(14, 1, 50)])
        groups = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)
        result = validity_known_groups(scores, groups)
        assert result["test"] == "ANOVA"
        assert result["n_groups"] == 3

    def test_effect_size_present(self, rng):
        scores = np.concatenate([rng.normal(10, 1, 50), rng.normal(15, 1, 50)])
        groups = np.array(["A"] * 50 + ["B"] * 50)
        result = validity_known_groups(scores, groups)
        assert np.isfinite(result["effect_size"])

    def test_single_group(self, rng):
        scores = rng.standard_normal(50)
        groups = np.array(["A"] * 50)
        result = validity_known_groups(scores, groups)
        assert result["test"] == "none"

    def test_group_means(self, rng):
        scores = np.concatenate([rng.normal(5, 0.1, 50), rng.normal(10, 0.1, 50)])
        groups = np.array(["lo"] * 50 + ["hi"] * 50)
        result = validity_known_groups(scores, groups)
        assert result["group_means"]["lo"] < result["group_means"]["hi"]
