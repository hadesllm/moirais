"""Test estimate_flops."""
from morie.fn.flops import estimate_flops, flops
from morie.fn._containers import DescriptiveResult


class TestEstimateFlops:
    def test_basic(self):
        result = estimate_flops(seq_len=512, d_model=768, n_layers=12, n_heads=12)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "estimate_flops"

    def test_positive(self):
        result = estimate_flops(seq_len=512, d_model=768, n_layers=12, n_heads=12)
        assert result.value > 0

    def test_scales_with_layers(self):
        r1 = estimate_flops(seq_len=512, d_model=768, n_layers=6, n_heads=12)
        r2 = estimate_flops(seq_len=512, d_model=768, n_layers=12, n_heads=12)
        assert r2.value > r1.value

    def test_alias(self):
        assert flops is estimate_flops
