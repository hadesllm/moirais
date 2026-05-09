"""Tests for moirais.fn.vmtmm — MTMM matrix analysis."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.vmtmm import validity_mtmm


class TestValidityMtmm:

    @pytest.fixture()
    def mtmm_data(self, rng):
        n = 100
        t1 = rng.standard_normal(n)
        t2 = rng.standard_normal(n)
        return pd.DataFrame({
            "t1_m1": t1 + rng.standard_normal(n) * 0.3,
            "t1_m2": t1 + rng.standard_normal(n) * 0.3,
            "t2_m1": t2 + rng.standard_normal(n) * 0.3,
            "t2_m2": t2 + rng.standard_normal(n) * 0.3,
        })

    def test_returns_dict(self, mtmm_data):
        traits = {"t1": ["t1_m1", "t1_m2"], "t2": ["t2_m1", "t2_m2"]}
        methods = {"m1": ["t1_m1", "t2_m1"], "m2": ["t1_m2", "t2_m2"]}
        result = validity_mtmm(mtmm_data, traits, methods)
        assert isinstance(result, dict)
        assert "correlation_matrix" in result

    def test_convergent_flag(self, mtmm_data):
        traits = {"t1": ["t1_m1", "t1_m2"], "t2": ["t2_m1", "t2_m2"]}
        methods = {"m1": ["t1_m1", "t2_m1"], "m2": ["t1_m2", "t2_m2"]}
        result = validity_mtmm(mtmm_data, traits, methods)
        assert isinstance(result["convergent_valid"], bool)

    def test_discriminant_flag(self, mtmm_data):
        traits = {"t1": ["t1_m1", "t1_m2"], "t2": ["t2_m1", "t2_m2"]}
        methods = {"m1": ["t1_m1", "t2_m1"], "m2": ["t1_m2", "t2_m2"]}
        result = validity_mtmm(mtmm_data, traits, methods)
        assert isinstance(result["discriminant_valid"], bool)
