"""Tests for moirais.fn.efa2 -- optimal number of factors."""

from moirais.fn.efa2 import efa_nfactors


class TestEfaNfactors:

    def test_parallel_analysis(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        result = efa_nfactors(mapq_df[items])
        assert result["n_factors"] >= 1
        assert result["method"] == "parallel"
        assert len(result["eigenvalues"]) == len(items)

    def test_map_method(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        result = efa_nfactors(mapq_df[items], method="map")
        assert result["n_factors"] >= 1
        assert result["method"] == "map"

    def test_bic_method(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        result = efa_nfactors(mapq_df[items], method="bic")
        assert result["n_factors"] >= 1
