"""Tests for moirais.fn.milat -- latent mean differences."""

from moirais.fn.milat import mi_latent_means


class TestMiLatentMeans:

    def test_returns_comparisons(self, mapq_df):
        result = mi_latent_means(mapq_df, "gender")
        assert "reference" in result
        assert "comparisons" in result
        assert len(result["comparisons"]) > 0

    def test_comparisons_have_cohens_d(self, mapq_df):
        result = mi_latent_means(mapq_df, "gender")
        for comp in result["comparisons"]:
            assert "cohens_d" in comp
            assert "p_value" in comp

    def test_four_factors_per_group(self, mapq_df):
        result = mi_latent_means(mapq_df, "gender")
        # 1 comparison group * 4 factors = 4
        assert len(result["comparisons"]) == 4
