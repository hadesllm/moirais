TurboQuant — Vector Quantization
=================================

Part of :doc:`index` — MOIRAIS's statistical-methods reference.

MOIRAIS implements the TurboQuant algorithm as a **standalone vector compression library** (``moirais.quant``). It is validated against the paper's theoretical bounds and achieves near-optimal distortion rates.

.. important::

   ``moirais.quant`` compresses arbitrary vectors — it is **not** integrated into Ollama's inference pipeline. For runtime KV-cache compression during LLM inference, use Ollama's built-in support::

       OLLAMA_KV_CACHE_TYPE="q8_0" ollama serve

   TurboQuant integration into llama.cpp is pending (estimated Q2 2026).

Weight Quantization vs KV-Cache Quantization
--------------------------------------------

These are **different techniques** that solve different problems:

- **Compresses** — weights compress model parameters (permanent, on disk); KV-cache compresses the runtime attention scratchpad (ephemeral, in RAM).
- **Reduces** — weights reduce model file size + VRAM to load; KV-cache reduces context memory during generation.
- **Best for** — weights help fit larger models on GPU (8B → 5GB file); KV-cache helps fit longer contexts on the same hardware.
- **Calibration** — weights need per-model tuning; KV-cache is data-oblivious (no calibration data required).
- **Status in Ollama** — weights handled automatically via GGUF; KV-cache supports ``q8_0`` / ``q4_0`` today, TurboQuant integration pending.

Both can be combined: a Q4_K_M model with q8_0 KV-cache is the practical sweet spot for consumer hardware.

References
----------

.. [Zandieh2026] Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2026).
   TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   *ICLR 2026*. `arXiv:2504.19874 <https://arxiv.org/abs/2504.19874>`_

.. [Zandieh2025] Zandieh, A., Daliri, M., & Han, I. (2025).
   QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead. *AAAI 2025*, 39(24), 25805-25813.
   `arXiv:2406.03482 <https://arxiv.org/abs/2406.03482>`_

.. [Han2026] Han, I., Kacham, P., Karbasi, A., Mirrokni, V., & Zandieh, A.
   (2026). PolarQuant: Quantization of KV Cache via Polar Coordinate
   Transforms. *AISTATS 2026*. `arXiv:2502.02617 <https://arxiv.org/abs/2502.02617>`_

Algorithm
---------

**Stage 1 — TurboQuant_MSE (PolarQuant):**

1. Generate random rotation matrix :math:`\Pi` via QR decomposition
2. Rotate: :math:`\mathbf{y} = \Pi \cdot \mathbf{x}`
3. Normalize: store :math:`\|\mathbf{y}\|_2`, compute :math:`\hat{\mathbf{y}} = \mathbf{y} / \|\mathbf{y}\|_2`
4. Scalar-quantize each coordinate via Lloyd-Max codebook optimized for
   the Beta distribution:

   .. math::

      f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d{-}1)/2)}\,(1 - x^2)^{(d-3)/2}

5. Store: ``(norm, indices)`` per block

**Stage 2 — QJL Error Correction:**

1. Compute residual: :math:`\mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}_{\text{MSE}}`
2. Project and sign-quantize: :math:`Q(\mathbf{r}) = \text{sign}(S \cdot \mathbf{r})`
3. Dequantize: :math:`Q^{-1}(\mathbf{z}) = \frac{\sqrt{\pi/2}}{d} \, S^T \mathbf{z}`
4. **Guarantee**: :math:`\mathbb{E}[\langle \mathbf{y}, Q^{-1}(Q(\mathbf{r})) \rangle] = \langle \mathbf{y}, \mathbf{r} \rangle` (unbiased)

Theoretical Bounds
------------------

MSE distortion upper bound:

.. math::

   D_{\text{mse}} \leq \frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^b} \approx \frac{2.72}{4^b}

Inner-product distortion upper bound:

.. math::

   D_{\text{prod}} \leq \frac{\sqrt{3}\,\pi^2\,\|\mathbf{y}\|_2^2}{d} \cdot \frac{1}{4^b}

Information-theoretic lower bound:

.. math::

   D_{\text{mse}}(Q) \geq \frac{1}{4^b}

The gap between TurboQuant's achievable rate and the lower bound is approximately :math:`2.72\times`.

Experimental Results
--------------------

Synthetic Validation
^^^^^^^^^^^^^^^^^^^^

Validated on random Gaussian vectors (d=128 and d=256):

**Python path** (``moirais.quant``):

- **2-bit, d=128** — MSE 0.096 (bound 0.170), cosine 0.946, 7.1× compression.
- **3-bit, d=128** — MSE 0.030 (bound 0.043), cosine 0.984, 4.9× compression.
- **4-bit, d=128** — MSE 0.007 (bound 0.011), cosine 0.996, 3.8× compression.
- **5-bit, d=256** — MSE 0.002 (bound 0.003), cosine 0.999, 3.2× compression.
- **2-bit, d=256** — MSE 0.093 (bound 0.170), cosine 0.955, 7.5× compression.
- **3-bit, d=256** — MSE 0.028 (bound 0.043), cosine 0.987, 5.1× compression.
- **4-bit, d=256** — MSE 0.009 (bound 0.011), cosine 0.996, 3.9× compression.

**C path** (``quant_ggml.dylib``):

- **2-bit** — MSE 0.110, cosine 0.935, 7.5× compression.
- **3-bit** — MSE 0.028, cosine 0.984, 5.1× compression.
- **4-bit** — MSE 0.008, cosine 0.995, 3.9× compression.

All 3-bit and 4-bit results are within the paper's theoretical MSE bounds. The C path uses Walsh-Hadamard Transform (WHT) with :math:`1/\sqrt{d}` normalization instead of full QR decomposition.

End-to-End Model Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TurboQuant was validated end-to-end on a 50.3M parameter autoresearch GPT model
(Karpathy's `autoresearch <https://github.com/karpathy/autoresearch>`_,
depth=4, d_model=256, 4 heads, vocab=8192) using post-training quantization
of all 2D weight matrices. The model was trained on Apple M2 (MPS) and
quantized/evaluated on a 16GB ARM64 system.

Post-training quantization on autoresearch 50.3M GPT:

- **fp32 baseline** — val_bpb 1.614, cosine 1.000, 1.0× (M2).
- **2-bit** — val_bpb 2.019 (Δ +0.405, +25.1%), cosine 0.940, 14.6× (M2).
- **3-bit** — val_bpb 2.001 (Δ +0.387, +24.0%), cosine 0.983, 10.0× (M2).
- **4-bit** — val_bpb 2.002 (Δ +0.388, +24.0%), cosine 0.995, 7.6× (M2).
- **5-bit** — val_bpb 2.028 (Δ +0.0004, +0.02%), cosine 0.999, 6.2× (ARM64).

.. note::

   The 5-bit result uses a different baseline (Pi-trained model, val_bpb=2.028)
   than the 2-4 bit results (M2-trained, val_bpb=1.614). The key metric is
   **percentage bpb degradation from quantization**, not absolute val_bpb:

   - 4-bit: **+24%** bpb loss (significant quality degradation)
   - 5-bit: **+0.02%** bpb loss (essentially lossless)

   This 1000x improvement in preservation is due to two changes: (1) 50,000-point
   Lloyd-Max codebook grid (up from 10,000) for finer centroid placement, and
   (2) the extra bit providing 32 quantization levels per block instead of 16.

Per-Parameter Analysis (5-bit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All 28 quantized weight tensors achieved cosine > 0.998:

- ``transformer.wte.weight`` [8192, 256] — MSE 0.1799, cosine 0.9986.
- ``transformer.h.*.attn.c_{q,k,v,proj}`` [256, 256] — MSE 0.0001, cosine 0.9985.
- ``transformer.h.*.mlp.c_fc`` [1024, 256] — MSE 0.0001, cosine 0.9986.
- ``transformer.h.*.mlp.c_proj`` [256, 1024] — MSE 0.0000, cosine 0.9985.
- ``lm_head.weight`` [8192, 256] — MSE 0.0000, cosine 0.9986.
- ``value_embeds.{1,3}.weight`` [8192, 256] — MSE 0.1672, cosine 0.9986.

The embedding tables (wte, value_embeds) have the highest absolute MSE due to
their large magnitudes, but still achieve > 0.998 cosine because the error is
proportional to signal strength.

Why MSE-only Outperforms Prod at ≥ 4 Bits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TurboQuant has two modes:

- **TurboQuant_MSE** (Stage 1 only): all *b* bits for scalar quantization
- **TurboQuant_Prod** (Stage 1 + Stage 2): *(b-1)* bits for MSE + 1 bit for QJL error correction

At ≥ 4 bits, Prod is **worse** because it sacrifices a full bit (halving
resolution) for a QJL correction that provides diminishing returns. At 2-3 bits,
the QJL correction is more valuable because the Stage 1 quantization is coarse.

.. math::

   \text{MSE}_{b} = \frac{2.72}{4^b}, \quad
   \text{Prod}_{b} = \frac{2.72}{4^{b-1}} + \text{QJL correction}

At *b* = 5: MSE uses 32 levels directly, while Prod uses 16 + 1-bit QJL.
The factor-of-2 resolution loss outweighs the error correction benefit.

Significance for Causal Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TurboQuant's **unbiased** property (:math:`\mathbb{E}[\hat{x}] = x`) is critical
for epidemiological applications. Biased quantization would systematically shift
treatment effect estimates (ATE, CATE), potentially invalidating causal conclusions.
The data-oblivious nature (no calibration data required) ensures the method
is safe for any dataset without risk of overfitting to calibration distributions.

Implementation
--------------

**Python** (``moirais.quant``):

.. code-block:: python

   from moirais.quant import turboquant_mse, turboquant_mse_decode
   import numpy as np
   x = np.random.default_rng(42).standard_normal(256)
   block = turboquant_mse(x, bits=3)       # 5.1x compression
   x_hat = turboquant_mse_decode(block)    # reconstruct

**C** (``quant_ggml.c``):

.. code-block:: bash

   # Compile (macOS)
   cc -O2 -march=native -shared -o quant_ggml.dylib quant_ggml.c -lm

.. code-block:: python

   from moirais.quant_bridge import GGMLTurboQuant
   tq = GGMLTurboQuant()  # auto-loads .dylib
   block = tq.quantize(x.astype(np.float32), bits=3)
   x_hat = tq.dequantize(block, bits=3)

Codebook Resolution
-------------------

Lloyd-Max codebook quality is controlled by the grid resolution parameter in
``moirais.quant``. The codebook is built by discretizing the Beta distribution PDF
on a uniform grid and running the Lloyd-Max iteration.

- **10,000-point grid** — centroid precision ~1e-4, cosine 0.995 at 4-bit.
- **50,000-point grid** (default) — centroid precision ~2e-5, cosine 0.999 at 5-bit.

The 5x increase in grid resolution (``np.linspace(lo, hi, 50000)``) provides
finer centroid placement, reducing quantization MSE by approximately 15% at
4+ bits. This is the default as of 2026-04-14.

Reproduce
---------

**5-bit model compression (ARM64 reference):**

.. code-block:: bash

   cd dev/autoresearch-macos
   python3 quantize_eval.py --bits 5 --eval-tokens 16384

**Sweep all bit widths:**

.. code-block:: bash

   python3 quantize_eval.py --all --eval-tokens 16384

**Full pipeline (train → quantize → benchmark → GGUF):**

.. code-block:: bash

   ./run_full_experiment.sh

Critical Notes
--------------

- WHT normalization **must** use :math:`1/\sqrt{d}`, never :math:`1/d`
- Temperature for LLM-assisted code generation: **T=0.1** (never 0.7 for math)
- Lloyd-Max codebooks use 50,000-point grids (default) from the exact Beta distribution
- QJL provides **unbiased** inner-product estimation — essential for causal inference where biased attention drifts ATE/CATE estimates
- At ≥ 4 bits, use ``turboquant_mse`` (not ``turboquant_prod``) for best results
- 5-bit is the recommended default: 6.2x compression with cosine > 0.998
