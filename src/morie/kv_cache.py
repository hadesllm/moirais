"""TurboQuant-compressed KV cache for MORIE's inference engine.

Stores attention key/value vectors as compressed TQBlocks instead of raw
float tensors, achieving 4-6x memory reduction during inference.

This plugs into :class:`morie.engine.MORIEEngine` to provide KV-cache
compression during actual transformer attention computation.

References
----------
* TurboQuant: Zandieh et al. (2026). ICLR 2026. arXiv:2504.19874
* QJL: Zandieh et al. (2025). AAAI 2025. arXiv:2406.03482
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .quant import TQBlock, turboquant_mse, turboquant_mse_decode

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]


@dataclass
class CacheStats:
    """Memory statistics for a TurboQuantKVCache."""

    compressed_bytes: int = 0
    uncompressed_bytes: int = 0
    n_tokens: int = 0
    n_layers: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 0.0
        return self.uncompressed_bytes / self.compressed_bytes

    @property
    def savings_mb(self) -> float:
        return (self.uncompressed_bytes - self.compressed_bytes) / (1024 * 1024)


class TurboQuantKVCache:
    """KV cache that stores keys and values as TurboQuant-compressed blocks.

    Each key/value vector is quantized via :func:`morie.quant.turboquant_mse`
    on ``append()``, and decompressed on ``get_keys()`` / ``get_values()``.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers.
    head_dim : int
        Dimension per attention head (must be power of 2).
    bits : int
        TurboQuant quantization bits (2, 3, or 4).
    rotation_seed : int
        Shared rotation seed for reproducibility.

    Examples
    --------
    >>> cache = TurboQuantKVCache(n_layers=32, head_dim=128, bits=3)
    >>> k = np.random.randn(128)
    >>> v = np.random.randn(128)
    >>> cache.append(layer=0, k_vec=k, v_vec=v)
    >>> keys = cache.get_keys(0)   # (1, 128) decompressed
    >>> values = cache.get_values(0)
    >>> cache.stats.compression_ratio
    5.1
    """

    def __init__(
        self,
        n_layers: int,
        head_dim: int,
        bits: int = 3,
        rotation_seed: int = 42,
    ):
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.bits = bits
        self.rotation_seed = rotation_seed

        # Per-layer lists of TQBlocks
        self._k_cache: list[list[TQBlock]] = [[] for _ in range(n_layers)]
        self._v_cache: list[list[TQBlock]] = [[] for _ in range(n_layers)]

    def append(self, layer: int, k_vec: F64, v_vec: F64) -> None:
        """Compress and cache a new key/value pair for one token.

        Parameters
        ----------
        layer : int
            Transformer layer index.
        k_vec : ndarray of shape (head_dim,)
            Key vector (will be quantized).
        v_vec : ndarray of shape (head_dim,)
            Value vector (will be quantized).
        """
        k_block = turboquant_mse(
            k_vec.astype(np.float64),
            bits=self.bits,
            rotation_seed=self.rotation_seed,
        )
        v_block = turboquant_mse(
            v_vec.astype(np.float64),
            bits=self.bits,
            rotation_seed=self.rotation_seed,
        )
        self._k_cache[layer].append(k_block)
        self._v_cache[layer].append(v_block)

    def get_keys(self, layer: int) -> F64:
        """Decompress all cached keys for a layer.

        Returns
        -------
        ndarray of shape (seq_len, head_dim)
        """
        if not self._k_cache[layer]:
            return np.zeros((0, self.head_dim))
        return np.stack([turboquant_mse_decode(b) for b in self._k_cache[layer]])

    def get_values(self, layer: int) -> F64:
        """Decompress all cached values for a layer.

        Returns
        -------
        ndarray of shape (seq_len, head_dim)
        """
        if not self._v_cache[layer]:
            return np.zeros((0, self.head_dim))
        return np.stack([turboquant_mse_decode(b) for b in self._v_cache[layer]])

    @property
    def seq_len(self) -> int:
        """Number of tokens currently cached (from layer 0)."""
        return len(self._k_cache[0]) if self._k_cache else 0

    def clear(self) -> None:
        """Clear all cached blocks."""
        for layer_k, layer_v in zip(self._k_cache, self._v_cache):
            layer_k.clear()
            layer_v.clear()

    @property
    def stats(self) -> CacheStats:
        """Compute memory statistics."""
        compressed = 0
        uncompressed = 0
        n_tokens = self.seq_len

        for layer in range(self.n_layers):
            for block in self._k_cache[layer]:
                compressed += block.total_bits // 8
                uncompressed += block.d * 2  # FP16 = 2 bytes per element
            for block in self._v_cache[layer]:
                compressed += block.total_bits // 8
                uncompressed += block.d * 2

        return CacheStats(
            compressed_bytes=compressed,
            uncompressed_bytes=uncompressed,
            n_tokens=n_tokens,
            n_layers=self.n_layers,
        )


class UncompressedKVCache:
    """Baseline uncompressed KV cache for comparison benchmarks."""

    def __init__(self, n_layers: int, head_dim: int):
        self.n_layers = n_layers
        self.head_dim = head_dim
        self._k_cache: list[list[F64]] = [[] for _ in range(n_layers)]
        self._v_cache: list[list[F64]] = [[] for _ in range(n_layers)]

    def append(self, layer: int, k_vec: F64, v_vec: F64) -> None:
        self._k_cache[layer].append(k_vec.copy())
        self._v_cache[layer].append(v_vec.copy())

    def get_keys(self, layer: int) -> F64:
        if not self._k_cache[layer]:
            return np.zeros((0, self.head_dim))
        return np.stack(self._k_cache[layer])

    def get_values(self, layer: int) -> F64:
        if not self._v_cache[layer]:
            return np.zeros((0, self.head_dim))
        return np.stack(self._v_cache[layer])

    @property
    def seq_len(self) -> int:
        return len(self._k_cache[0]) if self._k_cache else 0

    def clear(self) -> None:
        for layer_k, layer_v in zip(self._k_cache, self._v_cache):
            layer_k.clear()
            layer_v.clear()

    @property
    def memory_bytes(self) -> int:
        total = 0
        for layer in range(self.n_layers):
            total += len(self._k_cache[layer]) * self.head_dim * 8  # float64
            total += len(self._v_cache[layer]) * self.head_dim * 8
        return total
