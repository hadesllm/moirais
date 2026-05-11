"""Tests for morie.kv_cache — TurboQuant-compressed KV cache."""

import numpy as np
import pytest

from morie.kv_cache import CacheStats, TurboQuantKVCache, UncompressedKVCache


RNG = np.random.default_rng(42)


class TestCacheStats:
    def test_compression_ratio_zero(self):
        stats = CacheStats(compressed_bytes=0, uncompressed_bytes=100)
        assert stats.compression_ratio == 0.0

    def test_compression_ratio(self):
        stats = CacheStats(compressed_bytes=100, uncompressed_bytes=500)
        assert stats.compression_ratio == pytest.approx(5.0)

    def test_savings_mb(self):
        stats = CacheStats(compressed_bytes=0, uncompressed_bytes=1024 * 1024)
        assert stats.savings_mb == pytest.approx(1.0)


class TestTurboQuantKVCache:
    @pytest.fixture
    def cache(self):
        return TurboQuantKVCache(n_layers=4, head_dim=128, bits=3)

    def test_empty_cache(self, cache):
        assert cache.seq_len == 0
        keys = cache.get_keys(0)
        assert keys.shape == (0, 128)

    def test_append_and_retrieve(self, cache):
        k = RNG.standard_normal(128)
        v = RNG.standard_normal(128)
        cache.append(layer=0, k_vec=k, v_vec=v)

        assert cache.seq_len == 1
        keys = cache.get_keys(0)
        values = cache.get_values(0)
        assert keys.shape == (1, 128)
        assert values.shape == (1, 128)

    def test_cosine_similarity_after_roundtrip(self, cache):
        """Average cosine sim over many vectors should be high."""
        rng = np.random.default_rng(42)
        sims = []
        for _ in range(50):
            k = rng.standard_normal(128)
            v = rng.standard_normal(128)
            cache_local = TurboQuantKVCache(n_layers=1, head_dim=128, bits=3)
            cache_local.append(layer=0, k_vec=k, v_vec=v)
            k_hat = cache_local.get_keys(0)[0]
            cos_sim = np.dot(k, k_hat) / (np.linalg.norm(k) * np.linalg.norm(k_hat))
            sims.append(cos_sim)
        # Average cosine similarity for 3-bit d=128 should be > 0.8
        assert np.mean(sims) > 0.8

    def test_multi_token_multi_layer(self, cache):
        for token_idx in range(10):
            for layer in range(4):
                k = RNG.standard_normal(128)
                v = RNG.standard_normal(128)
                cache.append(layer=layer, k_vec=k, v_vec=v)

        assert cache.seq_len == 10
        for layer in range(4):
            assert cache.get_keys(layer).shape == (10, 128)
            assert cache.get_values(layer).shape == (10, 128)

    def test_compression_ratio(self, cache):
        for _ in range(16):
            k = RNG.standard_normal(128)
            v = RNG.standard_normal(128)
            cache.append(layer=0, k_vec=k, v_vec=v)

        stats = cache.stats
        assert stats.compressed_bytes > 0
        assert stats.uncompressed_bytes > stats.compressed_bytes
        assert stats.compression_ratio > 3.0  # 3-bit should give > 3x

    def test_clear(self, cache):
        cache.append(layer=0, k_vec=RNG.standard_normal(128), v_vec=RNG.standard_normal(128))
        assert cache.seq_len == 1
        cache.clear()
        assert cache.seq_len == 0

    def test_stats_fields(self, cache):
        cache.append(layer=0, k_vec=RNG.standard_normal(128), v_vec=RNG.standard_normal(128))
        stats = cache.stats
        assert stats.n_layers == 4
        assert stats.n_tokens == 1


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_different_bit_widths(bits):
    """Average cosine over many vectors should exceed threshold per bit width."""
    rng = np.random.default_rng(77)
    sims = []
    for _ in range(50):
        cache = TurboQuantKVCache(n_layers=1, head_dim=128, bits=bits)
        k = rng.standard_normal(128)
        v = rng.standard_normal(128)
        cache.append(layer=0, k_vec=k, v_vec=v)
        k_hat = cache.get_keys(0)[0]
        cos_sim = np.dot(k, k_hat) / (np.linalg.norm(k) * np.linalg.norm(k_hat))
        sims.append(cos_sim)
    mean_sim = np.mean(sims)
    if bits >= 3:
        assert mean_sim > 0.9
    else:
        assert mean_sim > 0.8


class TestUncompressedKVCache:
    def test_exact_roundtrip(self):
        cache = UncompressedKVCache(n_layers=2, head_dim=64)
        k = RNG.standard_normal(64)
        v = RNG.standard_normal(64)
        cache.append(layer=0, k_vec=k, v_vec=v)

        k_out = cache.get_keys(0)[0]
        np.testing.assert_array_equal(k_out, k)

    def test_memory_bytes(self):
        cache = UncompressedKVCache(n_layers=1, head_dim=64)
        cache.append(layer=0, k_vec=np.zeros(64), v_vec=np.zeros(64))
        # 2 vectors * 64 elements * 8 bytes (float64)
        assert cache.memory_bytes == 2 * 64 * 8

    def test_clear(self):
        cache = UncompressedKVCache(n_layers=1, head_dim=64)
        cache.append(layer=0, k_vec=np.zeros(64), v_vec=np.zeros(64))
        cache.clear()
        assert cache.seq_len == 0

    def test_empty(self):
        cache = UncompressedKVCache(n_layers=2, head_dim=32)
        assert cache.seq_len == 0
        assert cache.get_keys(0).shape == (0, 32)


class TestCompressedVsUncompressed:
    """Compare TurboQuant cache against uncompressed baseline."""

    def test_compressed_uses_less_memory(self):
        dim = 128
        n_layers = 4
        n_tokens = 32

        compressed = TurboQuantKVCache(n_layers=n_layers, head_dim=dim, bits=3)
        uncompressed = UncompressedKVCache(n_layers=n_layers, head_dim=dim)

        for _ in range(n_tokens):
            for layer in range(n_layers):
                k = RNG.standard_normal(dim)
                v = RNG.standard_normal(dim)
                compressed.append(layer=layer, k_vec=k, v_vec=v)
                uncompressed.append(layer=layer, k_vec=k, v_vec=v)

        stats = compressed.stats
        assert stats.compressed_bytes < uncompressed.memory_bytes
        assert stats.compression_ratio > 3.0
