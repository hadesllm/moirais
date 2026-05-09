"""Tests for moirais.fn.gwas1 — single-SNP GWAS."""
import numpy as np
import pytest
from moirais.fn.gwas1 import gwas_single_snp


class TestGWAS:
    def test_associated_snp(self):
        rng = np.random.default_rng(42)
        g = rng.choice([0, 1, 2], size=200)
        y = 0.5 * g + rng.standard_normal(200)
        res = gwas_single_snp(g, y)
        assert res.p_value < 0.05

    def test_null_snp(self):
        rng = np.random.default_rng(42)
        g = rng.choice([0, 1, 2], size=200)
        y = rng.standard_normal(200)
        res = gwas_single_snp(g, y)
        assert res.p_value > 0.01
