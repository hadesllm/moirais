"""MOIRAIS Inference Engine — Transformer forward pass with TurboQuant KV-cache.

Minimal LLM inference engine with two backends:

- **MLX** (Apple Silicon GPU via Metal) — requires ``pip install mlx``
  on Python 3.14. Use ``.venv-314/`` for MLX acceleration.
- **NumPy** (CPU fallback) — works on Python 3.15, any platform.

The engine loads GGUF models via :class:`moirais.gguf_loader.GGUFModel`,
compresses the KV-cache with :class:`moirais.kv_cache.TurboQuantKVCache`,
and tokenizes with :class:`moirais.tokenizer.Tokenizer`.

Architecture: Llama-family (RMSNorm, RoPE, GQA, SwiGLU FFN) and
autoresearch GPT (ReLU², value embeddings, residual lambdas, logit softcap).

References
----------
* Karpathy, A. (2023). llama2.c — inference in pure C.
* TurboQuant: Zandieh et al. (2026). ICLR 2026. arXiv:2504.19874
* MLX: Apple (2023). ml-explore/mlx — array framework for Apple Silicon.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gguf_loader import GGUFModel
from .kv_cache import TurboQuantKVCache
from .tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Backend detection: prefer MLX on Apple Silicon, fall back to NumPy
# ---------------------------------------------------------------------------
_BACKEND = "numpy"
_mx = None

try:
    import mlx.core as _mx_core

    _mx = _mx_core
    _BACKEND = "mlx"
except ImportError:
    pass


def backend() -> str:
    """Return the active compute backend: ``'mlx'`` or ``'numpy'``."""
    return _BACKEND


def _to_mx(arr: np.ndarray):
    """Convert NumPy array to MLX array."""
    return _mx.array(arr.astype(np.float32))


def _from_mx(arr) -> np.ndarray:
    """Convert MLX array to NumPy."""
    return np.array(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Math primitives — MLX-accelerated or NumPy fallback
# ---------------------------------------------------------------------------


def _rmsnorm(x, weight, eps: float = 1e-5):
    """RMS normalization: x * weight / rms(x)."""
    if _mx is not None:
        rms = _mx.sqrt(_mx.mean(x * x, axis=-1, keepdims=True) + eps)
        return x / rms * weight
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def _rmsnorm_bare(x, eps: float = 1e-5):
    """RMS normalization without learnable weight (autoresearch GPT style)."""
    if _mx is not None:
        rms = _mx.sqrt(_mx.mean(x * x, axis=-1, keepdims=True) + eps)
        return x / rms
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms


def _softmax(x):
    """Numerically stable softmax over last axis."""
    if _mx is not None:
        return _mx.softmax(x, axis=-1)
    x_max = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def _rope(q, k, position: int, head_dim: int, rope_freq_base: float = 10000.0):
    """Apply Rotary Position Embedding (RoPE) to query and key vectors.

    Always operates on NumPy arrays (per-head, not a matmul bottleneck).
    """
    half = head_dim // 2
    freqs = 1.0 / (rope_freq_base ** (np.arange(0, half, dtype=np.float32) / half))
    angles = position * freqs
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    def _rotate(v):
        v1 = v[..., :half]
        v2 = v[..., half:]
        return np.concatenate([v1 * cos_a - v2 * sin_a, v1 * sin_a + v2 * cos_a], axis=-1)

    return _rotate(q), _rotate(k)


def _silu(x):
    """SiLU activation: x * sigmoid(x)."""
    if _mx is not None:
        return x * _mx.sigmoid(x)
    return x * (1.0 / (1.0 + np.exp(-x)))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Container for generation output."""

    text: str
    tokens: list[int]
    n_tokens: int = 0
    time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    kv_compression_ratio: float = 0.0
    backend: str = "numpy"


class MOIRAISEngine:
    """MOIRAIS's own LLM inference engine with TurboQuant KV-cache compression.

    Parameters
    ----------
    model_path : str or Path
        Path to a GGUF model file.
    kv_bits : int
        TurboQuant quantization bits for KV-cache (2, 3, or 4).
    use_kv_compression : bool
        Whether to use TurboQuant KV-cache compression.

    Examples
    --------
    >>> engine = MOIRAISEngine("~/.ollama/models/blobs/sha256-abc123", kv_bits=3)
    >>> result = engine.generate("The capital of France is", max_tokens=20)
    >>> print(result.text)
    """

    def __init__(
        self,
        model_path: str | Path,
        kv_bits: int = 3,
        use_kv_compression: bool = True,
    ):
        self.model = GGUFModel(model_path)
        self.config = self.model.config
        self.tokenizer = Tokenizer(gguf_model=self.model)

        self._arch = self.config.get("architecture", "llama")
        self._is_moirais_gpt = self._arch == "moirais_gpt"

        n_layers = self.config.get("n_layers", 32)
        n_kv_heads = self.config.get("n_kv_heads", self.config.get("n_heads", 32))
        hidden_dim = self.config.get("hidden_dim", 4096)
        n_heads = self.config.get("n_heads", 32)
        head_dim = self.config.get("head_dim", hidden_dim // n_heads if n_heads else 128)

        n_cache_layers = n_layers * n_kv_heads

        self.use_kv_compression = use_kv_compression
        if use_kv_compression:
            self.kv_cache = TurboQuantKVCache(
                n_layers=n_cache_layers,
                head_dim=head_dim,
                bits=kv_bits,
            )
        else:
            from .kv_cache import UncompressedKVCache

            self.kv_cache = UncompressedKVCache(
                n_layers=n_cache_layers,
                head_dim=head_dim,
            )

        # Preload weights into MLX if available
        self._weight_cache: dict[str, object] = {}
        self._backend = _BACKEND

    def _get_weight(self, name: str):
        """Load a tensor, convert to MLX if available, cache result."""
        if name in self._weight_cache:
            return self._weight_cache[name]
        w = self.model.get_tensor(name)
        if _mx is not None:
            w = _to_mx(w)
        self._weight_cache[name] = w
        return w

    def forward(self, token_id: int, position: int):
        """One transformer forward pass — returns logits.

        Parameters
        ----------
        token_id : int
            Input token ID.
        position : int
            Sequence position (for RoPE).

        Returns
        -------
        ndarray
            Logits over vocabulary (vocab_size,).
        """
        embd = self._get_weight("token_embd.weight")
        x = embd[token_id]

        n_layers = self.config.get("n_layers", 32)
        hidden_dim = self.config.get("hidden_dim", 4096)
        n_heads = self.config.get("n_heads", 32)
        head_dim = self.config.get("head_dim", hidden_dim // n_heads if n_heads else 128)
        n_kv_heads = self.config.get("n_kv_heads", n_heads)
        rope_base = self.config.get("rope_freq_base", 10000.0)
        norm_eps = self.config.get("norm_eps", 1e-5)

        if self._is_moirais_gpt:
            x = _rmsnorm(x, np.ones_like(x), eps=norm_eps)
            x0 = x.copy()

        for layer in range(n_layers):
            if self._is_moirais_gpt:
                x, x0 = self._transformer_block_moirais(
                    x,
                    x0,
                    layer,
                    position,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    rope_base=rope_base,
                    eps=norm_eps,
                )
            else:
                x = self._transformer_block(
                    x,
                    layer,
                    position,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    rope_base=rope_base,
                    eps=norm_eps,
                )

        if self._is_moirais_gpt:
            x = _rmsnorm_bare(x, eps=norm_eps)
        else:
            norm_w = self._get_weight("output_norm.weight")
            x = _rmsnorm(x, norm_w, eps=norm_eps)

        output_w = self._get_weight("output.weight")
        if _mx is not None:
            logits = _from_mx(_to_mx(x) @ output_w.T)
        else:
            logits = x @ output_w.T

        if self._is_moirais_gpt:
            cap = self.config.get("logit_softcap", 15.0)
            if cap > 0:
                logits = cap * np.tanh(logits / cap)

        return logits

    def _transformer_block(
        self,
        x,
        layer: int,
        position: int,
        *,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_base: float,
        eps: float,
    ):
        """Attention + FFN with residual connections."""
        prefix = f"blk.{layer}"

        # Attention
        norm_w = self._get_weight(f"{prefix}.attn_norm.weight")
        normed = _rmsnorm(x, norm_w, eps=eps)
        attn_out = self._attention(
            normed,
            layer,
            position,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_base=rope_base,
        )
        x = x + attn_out

        # FFN
        ffn_norm_w = self._get_weight(f"{prefix}.ffn_norm.weight")
        normed = _rmsnorm(x, ffn_norm_w, eps=eps)
        ffn_out = self._ffn(normed, layer)
        return x + ffn_out

    def _attention(
        self,
        x,
        layer: int,
        position: int,
        *,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_base: float,
    ):
        """Multi-head attention with TurboQuant-compressed KV cache."""
        prefix = f"blk.{layer}"

        # Q, K, V projections
        wq = self._get_weight(f"{prefix}.attn_q.weight")
        wk = self._get_weight(f"{prefix}.attn_k.weight")
        wv = self._get_weight(f"{prefix}.attn_v.weight")

        if _mx is not None:
            q = x @ wq.T
            k = x @ wk.T
            v = x @ wv.T
            # Convert to numpy for KV-cache (TurboQuant is NumPy-based)
            q_np = _from_mx(q)
            k_np = _from_mx(k)
            v_np = _from_mx(v)
        else:
            q_np = x @ wq.T
            k_np = x @ wk.T
            v_np = x @ wv.T

        # Apply RoPE (per-head)
        # For GQA: q has n_heads * head_dim, k/v have n_kv_heads * head_dim
        q_heads = q_np.reshape(n_heads, head_dim)
        k_heads = k_np.reshape(n_kv_heads, head_dim)
        v_heads = v_np.reshape(n_kv_heads, head_dim)

        for h in range(n_heads):
            q_h = q_heads[h]
            # GQA: map query head to KV head
            kv_h = h * n_kv_heads // n_heads
            if h % (n_heads // n_kv_heads) == 0:
                k_h = k_heads[kv_h]
                q_r, k_r = _rope(q_h, k_h, position, head_dim, rope_base)
                k_heads[kv_h] = k_r
                # Store compressed K, V for this head
                self.kv_cache.append(
                    layer=layer * n_kv_heads + kv_h,
                    k_vec=k_r.astype(np.float64),
                    v_vec=v_heads[kv_h].astype(np.float64),
                )
            else:
                q_r, _ = _rope(q_h, q_h, position, head_dim, rope_base)
            q_heads[h] = q_r

        # Compute attention per head
        output_heads = []
        for h in range(n_heads):
            kv_h = h * n_kv_heads // n_heads
            cache_layer = layer * n_kv_heads + kv_h

            all_k = self.kv_cache.get_keys(cache_layer)  # (seq, head_dim)
            all_v = self.kv_cache.get_values(cache_layer)  # (seq, head_dim)

            q_h = q_heads[h]  # (head_dim,)
            scores = q_h @ all_k.T / math.sqrt(head_dim)  # (seq,)

            # Causal mask not needed for single-token generation
            # (all cached positions are valid)

            weights = np.exp(scores - np.max(scores))
            weights = weights / weights.sum()

            out_h = weights @ all_v  # (head_dim,)
            output_heads.append(out_h)

        attn_concat = np.concatenate(output_heads)  # (n_heads * head_dim,)

        # Output projection
        wo = self._get_weight(f"{prefix}.attn_output.weight")
        if _mx is not None:
            result = _to_mx(attn_concat) @ wo.T
        else:
            result = attn_concat @ wo.T

        return result

    def _transformer_block_moirais(
        self,
        x,
        x0,
        layer: int,
        position: int,
        *,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_base: float,
        eps: float,
    ):
        """Autoresearch GPT block: residual lambdas applied BEFORE the block,
        weightless RMSNorm, ReLU² FFN, value embeddings."""
        resid_lambdas = self._get_weight("resid_lambdas")
        x0_lambdas = self._get_weight("x0_lambdas")
        if _mx is not None:
            rl = float(_from_mx(resid_lambdas[layer]))
            xl = float(_from_mx(x0_lambdas[layer]))
        else:
            rl = float(resid_lambdas[layer])
            xl = float(x0_lambdas[layer])
        x = rl * x + xl * x0

        normed = _rmsnorm_bare(x, eps=eps)
        attn_out = self._attention(
            normed,
            layer,
            position,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_base=rope_base,
        )
        x = x + attn_out

        normed = _rmsnorm_bare(x, eps=eps)
        ffn_out = self._ffn_moirais(normed, layer)
        x = x + ffn_out

        return x, x0

    def _ffn(self, x, layer: int):
        """SwiGLU Feed-Forward Network (Llama)."""
        prefix = f"blk.{layer}"
        w_gate = self._get_weight(f"{prefix}.ffn_gate.weight")
        w_up = self._get_weight(f"{prefix}.ffn_up.weight")
        w_down = self._get_weight(f"{prefix}.ffn_down.weight")

        gate = _silu(x @ w_gate.T)
        up = x @ w_up.T
        return (gate * up) @ w_down.T

    def _ffn_moirais(self, x, layer: int):
        """ReLU-squared Feed-Forward Network (autoresearch GPT)."""
        prefix = f"blk.{layer}"
        w_up = self._get_weight(f"{prefix}.ffn_up.weight")
        w_down = self._get_weight(f"{prefix}.ffn_down.weight")

        if _mx is not None:
            hidden = _from_mx(_to_mx(x) @ w_up.T)
            activated = np.maximum(hidden, 0) ** 2
            return _from_mx(_to_mx(activated) @ w_down.T)
        else:
            hidden = x @ w_up.T
            activated = np.maximum(hidden, 0) ** 2
            return activated @ w_down.T

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Input text.
        max_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature (0.1 recommended for science).
        top_p : float
            Nucleus sampling threshold.

        Returns
        -------
        GenerationResult
        """
        # Reset KV-cache for fresh generation
        self.kv_cache.clear()

        tokens = self.tokenizer.encode(prompt)
        start_time = time.monotonic()
        gen_start = len(tokens)

        # Process prompt tokens
        for pos, tok in enumerate(tokens):
            logits = self.forward(tok, pos)

        # Generate new tokens
        for pos in range(gen_start, gen_start + max_tokens):
            # Sample next token
            next_token = self._sample(logits, temperature, top_p)
            tokens.append(next_token)

            if next_token == self.tokenizer.eos_id:
                break

            logits = self.forward(next_token, pos)

        elapsed = time.monotonic() - start_time
        n_generated = len(tokens) - gen_start

        # KV-cache stats
        kv_ratio = 0.0
        if self.use_kv_compression and hasattr(self.kv_cache, "stats"):
            stats = self.kv_cache.stats
            kv_ratio = stats.compression_ratio

        return GenerationResult(
            text=self.tokenizer.decode(tokens),
            tokens=tokens,
            n_tokens=n_generated,
            time_seconds=elapsed,
            tokens_per_second=n_generated / elapsed if elapsed > 0 else 0,
            kv_compression_ratio=kv_ratio,
            backend=self._backend,
        )

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float, top_p: float) -> int:
        """Sample a token from logits with temperature and nucleus sampling."""
        if temperature <= 0:
            return int(np.argmax(logits))

        probs = np.exp((logits - np.max(logits)) / temperature)
        probs = probs / probs.sum()

        # Top-p (nucleus) sampling
        sorted_idx = np.argsort(-probs)
        cumsum = np.cumsum(probs[sorted_idx])
        mask = cumsum <= top_p
        # Always keep at least one token
        mask[0] = True
        candidates = sorted_idx[mask]
        cand_probs = probs[candidates]
        cand_probs = cand_probs / cand_probs.sum()

        return int(np.random.choice(candidates, p=cand_probs))

    def reset(self) -> None:
        """Clear the KV-cache for a new generation."""
        self.kv_cache.clear()

    def close(self) -> None:
        """Release resources."""
        self._weight_cache.clear()
        self.model.close()

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"MOIRAISEngine(arch={cfg.get('architecture')}, "
            f"layers={cfg.get('n_layers')}, "
            f"heads={cfg.get('n_heads')}, "
            f"backend={self._backend})"
        )
