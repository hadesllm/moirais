# SPDX-License-Identifier: GPL-2.0-only
# Py↔R agreement smoke tests for the LLM-architecture suite.
# Source from the r-package root:  source("R/llm_arch.R"); source("tests/test_llm_arch_pyr_parity.R")

suppressPackageStartupMessages({})
# Adjust path if running from a different CWD.
.llm_path <- file.path("r-package", "morie", "R", "llm_arch.R")
if (!file.exists(.llm_path))
  .llm_path <- file.path("..", "r-package", "morie", "R", "llm_arch.R")
source(.llm_path)
`%||%` <- function(a, b) if (is.null(a)) b else a

tol <- 1e-6

# rmsnr
x <- matrix(c(3, 4, 1, 2), nrow = 2, byrow = TRUE)
r_R <- rms_norm(x, eps = 0)$tensor
# Python:  x / sqrt(mean(x^2, axis=-1, keepdims=True))
py_expected <- x / sqrt(rowMeans(x * x))
stopifnot(max(abs(r_R - py_expected)) < tol)

# tmpsc
z <- c(1, 2, 3)
r_R <- temperature_scaling(z, T = 1)$tensor
stopifnot(abs(sum(r_R) - 1) < tol)
e <- exp(z - max(z)); py_expected <- e / sum(e)
stopifnot(max(abs(r_R - py_expected)) < tol)

# topkd
r_R <- top_k_decoding(c(1, 2, 3, 4, 5), k = 2L)$tensor
stopifnot(sum(r_R > 0) == 2L)
stopifnot(abs(sum(r_R) - 1) < tol)

# toppd
r_R <- top_p_nucleus(c(0, 0, 5), p = 0.5)
stopifnot(r_R$n_kept == 1L)

# rptpn
r_R <- repetition_penalty(c(2, -2, 1), generated = c(0, 1), alpha = 2)$tensor
stopifnot(max(abs(r_R - c(1, -4, 1))) < tol)

# pplxm
stopifnot(abs(perplexity_metric(c(log(0.5), log(0.5)))$value - 2) < tol)

# bpblm
stopifnot(abs(bits_per_byte(rep(log(2), 4), n_bytes = 4)$value - 1) < tol)

# cslnc
stopifnot(abs(cosine_lr_schedule(0, lr_max = 1, lr_min = 0,
                                  total_steps = 10, warmup_steps = 0)$value - 1) < tol)

# grdcl
r_R <- gradient_clipping(c(3, 4), max_norm = 1)$tensor
stopifnot(abs(sqrt(sum(r_R * r_R)) - 1) < tol)

# lradw
stopifnot(abs(lr_warmup(500, lr_target = 1, warmup_steps = 1000)$value - 0.5) < tol)

# rlhfd
stopifnot(abs(rlhf_reward(matrix(c(1, 1), 1, 2), w = c(0.5, 0.5))$value - 1) < tol)

# flshA  (vs naive softmax(QK^T/sqrt(d))V)
set.seed(0)
Q <- matrix(rnorm(24), 6, 4); K <- matrix(rnorm(24), 6, 4); V <- matrix(rnorm(24), 6, 4)
out_flash <- flash_attention(Q, K, V, block_size = 2L)$tensor
s <- (Q %*% t(K)) / sqrt(4)
p <- exp(s - apply(s, 1, max)); p <- p / rowSums(p)
naive <- p %*% V
stopifnot(max(abs(out_flash - naive)) < 1e-9)

# spqkv  diagonal always on
stopifnot(all(diag(sparse_attention(8L, window = 1L,
                                     stride = 4L, n_random = 0L)$boolean)))

# moeml  routes argmax expert
W_gate <- matrix(c(10, 0, -10, 0), 2, 2)
stopifnot(mixture_of_experts(matrix(c(1, 0), 1, 2),
                              W_gate = W_gate, top_k = 1L)$topk_idx[1, 1] == 0L)

# kvcmp
r_R <- kv_cache_management(matrix(0, 2, 4), matrix(0, 2, 4),
                             matrix(1, 1, 4), matrix(1, 1, 4))
stopifnot(r_R$T == 3L)

# cslat
stopifnot(is.infinite(causal_attention_mask(3L)$tensor[1, 2]))

# wdemb
stopifnot(all.equal(word_embedding(c(0, 2), E = diag(4))$tensor,
                    diag(4)[c(1, 3), ]))

# tknbp
stopifnot(bpe_tokenizer(c("low", "low", "lower", "newest", "newest", "newest"),
                        num_merges = 3L)$n_merges == 3L)

# swigl  zero input -> zero output (SiLU(0)=0)
stopifnot(all(abs(swiglu_activation(matrix(0, 1, 4),
                                     W = diag(4), V = diag(4))$tensor) < tol))

# grpqa  shape check
Q <- array(0, c(4, 2, 8)); K <- V <- array(0, c(2, 2, 8))
stopifnot(identical(dim(grouped_query_attention(Q, K, V,
                                                  n_heads = 4L, n_kv_heads = 2L)$tensor),
                    c(4L, 2L, 8L)))

cat("ALL 20 R checks pass within tol=", tol, "\n", sep = "")
