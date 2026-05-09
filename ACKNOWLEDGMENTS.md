# Acknowledgments

MOIRAIS (Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation) is developed by Vansh Singh Ruhela via the Collaborative Specialization in Addiction Studies (CoPAS) between Dalla Lana School of Public Health (DLSPH) and the Centre for Criminology and Sociolegal Studies (CrimSL) at the University of Toronto School of Graduate Studies (UTSGS).

## AI Development Partners

**Anthropic / Claude** — Claude (Opus, Sonnet) serves as the AI co-architect of MOIRAIS, contributing to code generation, statistical function design, testing infrastructure, documentation, and architectural decisions across 2000+ function implementations. Claude Code is integral to the MOIRAIS development workflow.

**Google / Gemma** — The Gemma model family (Gemma 3, Gemma 4) powers Perseus, MOIRAIS's resident AI agent. Perseus (`perseus:e2b`) is a custom-tuned Gemma 4 model (7.2GB, Q4_K_M) with domain-specific expertise in causal inference, scientific experimentation, and statistical computing. Google's open model weights enable fully local, private AI inference.

## Frameworks and Tools

**Andrej Karpathy / autoresearch** — The [autoresearch](https://github.com/karpathy/autoresearch) framework provided the foundation for MOIRAIS's autonomous LLM pretraining experiments, including the 50.3M parameter model trained on Apple Silicon (M2) with MPS acceleration.

**TurboQuant** (Ankush Agarwal et al., ICLR 2026) — The TurboQuant algorithm (arxiv.org/abs/2504.19874) is implemented in `moirais.quant` for data-oblivious, unbiased KV-cache compression. PolarQuant + QJL + Lloyd-Max codebooks achieve 4-bit compression at 0.995 cosine similarity.

**Ollama** — [Ollama](https://ollama.com) provides the local LLM serving infrastructure that powers Perseus on both macOS and Raspberry Pi. Ollama's model management, quantization support, and simple API make local AI inference accessible across platforms.

**OllamaFreeAPI** — Community-maintained free API providing access to 16+ LLM models without API keys, serving as MOIRAIS's fallback provider when local models are unavailable.

## Open Source Dependencies

MOIRAIS builds on the work of many open-source projects, including but not limited to: NumPy, SciPy, pandas, scikit-learn, Textual, httpx, Sphinx, and the broader Python/R scientific computing ecosystem.

## Funding

This work is conducted with zero external funding, using exclusively free-tier and open-source tools.
