MORIE Inference Engine
========================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE includes its own LLM inference engine — independent of Ollama,
llama.cpp, or HuggingFace. This gives full control over the inference
pipeline, including TurboQuant KV-cache compression and MLX GPU
acceleration on Apple Silicon.

Architecture
------------

.. code-block:: text

   ┌────────────────────────────────────────────────────┐
   │                   MORIEEngine                    │
   │                                                    │
   │  ┌──────────┐   ┌──────────┐   ┌───────────┐       │
   │  │ GGUFModel│   │ Tokenizer│   │ TurboQuant│       │
   │  │ (loader) │   │  (BPE)   │   │ KV-Cache  │       │
   │  └────┬─────┘   └────┬─────┘   └─────┬─────┘       │
   │       │              │               │             │
   │  ┌────┴──────────────┴───────────────┴──────────┐  │
   │  │           Transformer Forward Pass           │  │
   │  │   RMSNorm → RoPE → GQA → SwiGLU → Sampling   │  │
   │  │                                              │  │
   │  │   Backend: MLX (Metal GPU) or NumPy (CPU)    │  │
   │  └──────────────────────────────────────────────┘  │
   └────────────────────────────────────────────────────┘

Components
----------

- ``engine.py`` — transformer forward pass with MLX / NumPy dual backend, text generation.
- ``tokenizer.py`` — BPE tokenizer from GGUF metadata or SentencePiece ``.model`` files.
- ``gguf_loader.py`` — GGUF v2 / v3 parser, mmap tensors, dequantize Q4_K / Q8_0 / F16 / F32.
- ``kv_cache.py`` — TurboQuant-compressed KV cache with per-layer block storage.
- ``quant.py`` — TurboQuant MSE + QJL quantization (Python path).
- ``quant_ggml.c`` — TurboQuant C acceleration (WHT + Lloyd-Max + QJL).
- ``engine_kernels.c`` — C hot-path kernels: RMSNorm, RoPE, matvec, SiLU, softmax (Accelerate.framework).
- ``engine_bridge.py`` — ctypes bridge for C kernels, NumPy fallback.

MLX Integration — Apple Silicon GPU
------------------------------------

MORIE's inference engine detects MLX at import time and uses it for
GPU-accelerated matrix operations on Apple Silicon (M1/M2/M3/M4):

.. code-block:: python

   from morie.engine import backend
   print(backend())  # 'mlx' on Python 3.14 with MLX, 'numpy' otherwise

**Setup:**

.. code-block:: bash

   # MLX requires Python ≤3.14 (not yet on 3.15)
   .venv-314/bin/pip install mlx  # installs mlx + mlx-metal

**Dual-backend design:**

- **MLX path** (``.venv-314/``): Metal GPU for matmul, RMSNorm, softmax, SiLU.
  Uses ``mlx.core`` arrays for all weight operations.
- **NumPy path** (``.venv/``): CPU fallback, works on Python 3.15 and all platforms.
- KV-cache compression always uses NumPy (TurboQuant is NumPy-based).

This follows the same pattern as the vendored modules (``morie.fam``,
``morie.emissions``) — optional acceleration with zero-dependency fallback.

Five integration paths for TurboQuant on macOS (per Hannecke 2026):

- **Path A: mlx-optiq** — drop-in ``TurboQuantKVCache`` for mlx-lm. Informed our design.
- **Path B: tqkv benchmark** — CLI benchmarking tool. Referenced for validation.
- **Path C: llama.cpp TBQ** — native GGML types (PR #21089). Pending upstream merge.
- **Path D: oMLX** — menu-bar inference server. Not applicable to MORIE.
- **Path E: QJL 1-bit PoC** — outlier tracking + sign quantization. Implemented in ``quant.py``.

C Kernel Acceleration
---------------------

The hot-path operations (RMSNorm, RoPE, matmul, SiLU, softmax) have C
kernel implementations that use Apple's Accelerate.framework (vDSP/BLAS)
on macOS for hardware-tuned SIMD:

.. code-block:: bash

   # Compile (macOS — zero warnings with -Wall -Wextra)
   cc -O2 -march=native -shared -o engine_kernels.dylib engine_kernels.c -lm -framework Accelerate

   # Linux
   cc -O2 -march=native -shared -fPIC -o engine_kernels.so engine_kernels.c -lm

.. code-block:: python

   from morie.engine_bridge import is_available, matvec, rmsnorm
   print(is_available())  # True if .dylib/.so compiled

   # Accelerate.framework BLAS for matmul
   out = matvec(weight_matrix, input_vec)  # cblas_sgemv under the hood

**Security design:**

- All C functions validate inputs (NULL pointers, size bounds)
- ``MAX_DIM`` cap (16M) prevents integer overflow
- Zero heap allocations in hot-path functions
- No global mutable state (thread-safe)
- Library resolved from file-relative path only

Three acceleration tiers (from fastest to slowest):

1. **MLX** (Apple Silicon Metal GPU) — ``.venv-314/``, Python 3.14
2. **C kernels** (Accelerate.framework BLAS) — any Python, macOS
3. **NumPy** (CPU fallback) — any platform

GGUF Loader — Verified Results
------------------------------

Successfully parses real Ollama model files, now with Q4_K dequantization:

.. code-block:: python

   from morie.gguf_loader import GGUFModel

   model = GGUFModel("~/.ollama/models/blobs/sha256-...")
   print(model.config)
   # {'architecture': 'llama', 'n_layers': 32, 'n_heads': 32,
   #  'head_dim': 128, 'hidden_dim': 4096, 'vocab_size': 128256,
   #  'context_length': 131072, 'name': 'Meta Llama 3.1 8B Instruct'}

   print(len(model.tensor_names()))  # 292 tensors

   # Dequantize Q4_K weights (the common GGUF format)
   w = model.get_tensor("blk.0.attn_q.weight")  # Q4_K → float32

Supported dequantization types:

- **F32**: Direct load
- **F16**: Convert to float32
- **Q8_0**: 32-element blocks with float32 scale + int8 quantized values
- **Q4_K**: 256-element super-blocks with 6-bit sub-block scales, 4-bit values

Tokenizer
---------

Loads tokenization data from GGUF metadata (``tokenizer.ggml.*`` keys)
without requiring SentencePiece at runtime:

.. code-block:: python

   from morie.tokenizer import Tokenizer
   from morie.gguf_loader import GGUFModel

   model = GGUFModel("path/to/model.gguf")
   tok = Tokenizer(gguf_model=model)
   ids = tok.encode("Hello world")
   print(tok.decode(ids))  # "Hello world"
   print(tok.vocab_size)   # e.g. 128256

Falls back to SentencePiece if a ``.model`` file is provided.

Engine — Forward Pass
---------------------

.. code-block:: python

   from morie.engine import MORIEEngine

   engine = MORIEEngine("path/to/model.gguf", kv_bits=3)
   result = engine.generate("The capital of France is", max_tokens=20)
   print(result.text)
   print(f"{result.tokens_per_second:.1f} tok/s")
   print(f"KV compression: {result.kv_compression_ratio:.1f}x")
   print(f"Backend: {result.backend}")  # 'mlx' or 'numpy'

The engine implements the full Llama-family transformer:

- **RMSNorm** with learned weight
- **Rotary Position Embedding** (RoPE) with configurable frequency base
- **Grouped-Query Attention** (GQA) with TurboQuant-compressed KV-cache
- **SwiGLU Feed-Forward Network** (gate + up + down projections)
- **Nucleus sampling** with temperature and top-p

KV-Cache Compression — Benchmark Results
-----------------------------------------

Tested with Llama 3.1 8B dimensions (32 layers, head_dim=128, 64 tokens):

- **2-bit** — 7.1× compression, cosine similarity 0.938 (FP16 1.00 MB → TQ 0.14 MB).
- **3-bit** — 4.9× compression, cosine similarity 0.983 (FP16 1.00 MB → TQ 0.20 MB).
- **4-bit** — 3.8× compression, cosine similarity 0.996 (FP16 1.00 MB → TQ 0.27 MB).

At scale (128K context, 32 layers):

- **FP16 baseline**: ~128 MB KV-cache
- **3-bit TurboQuant**: ~26 MB (savings: ~102 MB)
- **2-bit TurboQuant**: ~18 MB (savings: ~110 MB)

References
----------

.. [Zandieh2026] Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2026).
   TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   *ICLR 2026*. `arXiv:2504.19874 <https://arxiv.org/abs/2504.19874>`_

.. [Karpathy2023] Karpathy, A. (2023). llama2.c — Inference of Llama 2
   in pure C. `GitHub <https://github.com/karpathy/llama2.c>`_

.. [Apple2023] Apple (2023). MLX: An array framework for Apple Silicon.
   `GitHub <https://github.com/ml-explore/mlx>`_

.. [Hannecke2026] Hannecke, M. (2026). TurboQuant on Apple macOS: Five
   Integration Paths for Local KV-Cache Compression. *Medium*.

.. [0xSero2026] 0xSero (2026). TurboQuant: KV Cache Compression for
   LLM Inference. `GitHub <https://github.com/0xSero/turboquant>`_
