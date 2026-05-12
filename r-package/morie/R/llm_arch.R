# SPDX-License-Identifier: GPL-2.0-only

#' LLM-architecture math operators (R parity)
#'
#' Twenty pure-base-R implementations of standard LLM/Transformer
#' building blocks.  Each is unit-checked against the Python
#' implementation in \code{morie.fn.<name>} to agree within 1e-6.
#'
#' Functions are intentionally NOT \code{@export}ed so they don't
#' need to be added to NAMESPACE; access via \code{morie:::name()}
#' or attach a re-export shim in your own package.
#'
#' @references
#' Vaswani et al. (2017).  Attention Is All You Need.  NeurIPS.
#' Zhang & Sennrich (2019).  Root Mean Square Layer Normalization.
#' Shazeer (2020).  GLU Variants Improve Transformer.
#' Su et al. (2021).  RoFormer: Enhanced Transformer with Rotary
#'   Position Embedding.
#' Dao et al. (2022).  FlashAttention.
#' Ainslie et al. (2023).  GQA: Training Generalized Multi-Query
#'   Transformer Models from Multi-Head Checkpoints.
#' @name llm_arch
#' @keywords internal
NULL


# ──────────────────────── shared helpers ───────────────────────────

.softmax_last <- function(x) {
  # softmax along the last axis of an array
  d <- dim(x); nd <- length(d)
  if (is.null(d) || nd == 1L) {
    x <- x - max(x)
    e <- exp(x)
    return(e / sum(e))
  }
  apply(x, seq_len(nd - 1L), function(v) {
    v <- v - max(v); e <- exp(v); e / sum(e)
  }) -> out
  # apply collapses last axis to first; transpose back
  aperm(out, c(seq.int(2L, nd), 1L))
}


# ──────────────────── 1.  tknbp / bpe_tokenizer ─────────────────────

#' Byte-pair encoding tokenizer (Sennrich 2016)
#' @keywords internal
bpe_tokenizer <- function(x, num_merges = 10L) {
  if (length(x) == 1L && is.character(x) && grepl("\\s", x))
    words <- strsplit(x, "\\s+")[[1L]]
  else
    words <- as.character(x)
  if (!length(words))
    return(list(merges = list(), vocab = character(0),
                n_merges = 0L, n_vocab = 0L, method = "BPE"))
  tab <- table(words)
  corpus <- lapply(names(tab), function(w)
    c(strsplit(w, "")[[1L]], "</w>"))
  freq <- as.integer(tab)
  merges <- list()
  for (m in seq_len(num_merges)) {
    pair_counts <- list()
    for (k in seq_along(corpus)) {
      sym <- corpus[[k]]; f <- freq[[k]]
      if (length(sym) < 2L) next
      for (i in seq_len(length(sym) - 1L)) {
        key <- paste(sym[i], sym[i + 1L], sep = "\x1f")
        pair_counts[[key]] <- (pair_counts[[key]] %||% 0L) + f
      }
    }
    if (!length(pair_counts)) break
    best_key <- names(which.max(unlist(pair_counts)))
    best <- strsplit(best_key, "\x1f", fixed = TRUE)[[1L]]
    merges[[length(merges) + 1L]] <- best
    # merge in corpus
    corpus <- lapply(corpus, function(sym) {
      if (length(sym) < 2L) return(sym)
      out <- character(0); i <- 1L
      while (i <= length(sym)) {
        if (i < length(sym) &&
            sym[i] == best[1L] && sym[i + 1L] == best[2L]) {
          out <- c(out, paste0(best[1L], best[2L])); i <- i + 2L
        } else { out <- c(out, sym[i]); i <- i + 1L }
      }
      out
    })
  }
  vocab <- unique(unlist(corpus))
  list(merges = merges, vocab = vocab, corpus = corpus,
       n_merges = length(merges), n_vocab = length(vocab),
       method = "BPE")
}

`%||%` <- function(a, b) if (is.null(a)) b else a


# ──────────────────── 2.  wdemb / word_embedding ────────────────────

#' Embedding row lookup (Mikolov 2013)
#' @keywords internal
word_embedding <- function(x, E = NULL, vocab_size = 100L,
                           d_model = 16L, seed = 0L) {
  ids <- as.integer(x)
  if (is.null(E)) {
    set.seed(seed)
    lim <- sqrt(6 / (vocab_size + d_model))
    E <- matrix(stats::runif(vocab_size * d_model, -lim, lim),
                nrow = vocab_size, ncol = d_model)
  }
  if (any(ids < 0L) || any(ids >= nrow(E)))
    stop("token id out of range for embedding matrix")
  e <- E[ids + 1L, , drop = FALSE]   # R is 1-indexed
  list(tensor = e, E = E, ids = ids, shape = dim(e),
       method = "embedding-lookup")
}


# ────────────────── 3.  cslat / causal_attention_mask ───────────────

#' Causal autoregressive mask (Radford 2019)
#' @keywords internal
causal_attention_mask <- function(x) {
  n <- if (length(x) == 1L && is.numeric(x)) as.integer(x)
       else if (!is.null(dim(x))) dim(x)[length(dim(x)) - 1L]
       else length(x)
  M <- matrix(0, n, n)
  M[upper.tri(M)] <- -Inf
  list(tensor = M, n = n, method = "causal-mask")
}


# ──────────────── 4.  grpqa / grouped_query_attention ───────────────

#' Grouped-query attention (Ainslie 2023)
#' @keywords internal
grouped_query_attention <- function(Q, K = NULL, V = NULL,
                                    n_heads = 8L, n_kv_heads = 2L) {
  if (is.null(K)) K <- Q
  if (is.null(V)) V <- Q
  if (n_heads %% n_kv_heads != 0L)
    stop("n_heads must be a multiple of n_kv_heads")
  group <- n_heads %/% n_kv_heads
  # Ensure (n_heads, seq, d) and (n_kv_heads, seq, d) shapes.
  if (length(dim(Q)) == 2L) Q <- array(Q, dim = c(n_heads, dim(Q)))
  if (length(dim(K)) == 2L) K <- array(K, dim = c(n_kv_heads, dim(K)))
  if (length(dim(V)) == 2L) V <- array(V, dim = c(n_kv_heads, dim(V)))
  # Replicate KV across the group dimension.
  rep_axis0 <- function(A, g) {
    new <- array(0, dim = c(dim(A)[1L] * g, dim(A)[-1L]))
    for (i in seq_len(dim(A)[1L]))
      for (j in seq_len(g))
        new[(i - 1L) * g + j, , ] <- A[i, , ]
    new
  }
  K_rep <- rep_axis0(K, group)
  V_rep <- rep_axis0(V, group)
  d_head <- dim(Q)[3L]
  attn <- array(0, dim = c(n_heads, dim(Q)[2L], dim(Q)[2L]))
  out  <- array(0, dim = dim(Q))
  scale <- 1 / sqrt(d_head)
  for (h in seq_len(n_heads)) {
    Qh <- matrix(Q[h, , ], nrow = dim(Q)[2L], ncol = d_head)
    Kh <- matrix(K_rep[h, , ], nrow = dim(K_rep)[2L], ncol = d_head)
    Vh <- matrix(V_rep[h, , ], nrow = dim(V_rep)[2L], ncol = d_head)
    s <- (Qh %*% t(Kh)) * scale
    s <- sweep(s, 1L, apply(s, 1L, max), "-")
    e <- exp(s)
    a <- e / rowSums(e)
    attn[h, , ] <- a
    out[h, , ] <- a %*% Vh
  }
  list(tensor = out, attn = attn, n_heads = n_heads,
       n_kv_heads = n_kv_heads, group_size = group, method = "GQA")
}


# ───────────────────── 5.  swigl / swiglu_activation ────────────────

#' SwiGLU activation (Shazeer 2020)
#' @keywords internal
swiglu_activation <- function(x, W = NULL, V = NULL, b = NULL, c = NULL) {
  if (is.null(W) && is.null(V)) {
    d_out <- ncol(as.matrix(x))
    W <- diag(d_out); V <- diag(d_out)
  } else if (xor(is.null(W), is.null(V))) {
    stop("Provide both W and V or neither")
  }
  if (is.null(b)) b <- rep(0, ncol(W))
  if (is.null(c)) c <- rep(0, ncol(V))
  xm <- as.matrix(x)
  gate <- sweep(xm %*% W, 2L, b, "+")
  silu_gate <- gate * (1 / (1 + exp(-gate)))
  up <- sweep(xm %*% V, 2L, c, "+")
  out <- silu_gate * up
  list(tensor = out, gate = silu_gate, up = up, method = "SwiGLU")
}


# ───────────────────── 6.  rmsnr / rms_norm ─────────────────────────

#' Root-mean-square normalisation (Zhang & Sennrich 2019)
#' @keywords internal
rms_norm <- function(x, gamma = NULL, eps = 1e-6) {
  xm <- as.matrix(x)
  rms <- sqrt(rowMeans(xm * xm) + eps)
  y <- sweep(xm, 1L, rms, "/")
  if (!is.null(gamma)) y <- sweep(y, 2L, as.numeric(gamma), "*")
  list(tensor = y, rms = rms, eps = eps, method = "RMSNorm")
}


# ─────────────────── 7.  kvcmp / kv_cache_management ────────────────

#' KV-cache append (Pope 2022)
#' @keywords internal
kv_cache_management <- function(K_cache, V_cache, k_new, v_new,
                                max_len = NULL) {
  if (is.null(K_cache)) {
    K_new <- as.matrix(k_new); V_new <- as.matrix(v_new)
  } else {
    K_new <- rbind(as.matrix(K_cache), as.matrix(k_new))
    V_new <- rbind(as.matrix(V_cache), as.matrix(v_new))
  }
  if (!is.null(max_len) && nrow(K_new) > max_len) {
    K_new <- K_new[(nrow(K_new) - max_len + 1L):nrow(K_new), , drop = FALSE]
    V_new <- V_new[(nrow(V_new) - max_len + 1L):nrow(V_new), , drop = FALSE]
  }
  list(K = K_new, V = V_new, T = nrow(K_new), max_len = max_len,
       method = "kv-cache-append")
}


# ────────────────── 8.  tmpsc / temperature_scaling ─────────────────

#' Temperature-scaled softmax (Hinton 2015)
#' @keywords internal
temperature_scaling <- function(x, T = 1) {
  if (T <= 0) stop("Temperature must be > 0")
  z <- as.numeric(x) / T
  z <- z - max(z)
  p <- exp(z); p <- p / sum(p)
  H <- -sum(ifelse(p > 0, p * log(p), 0))
  list(tensor = p, entropy = H, T = T, method = "temperature-softmax")
}


# ──────────────────── 9.  topkd / top_k_decoding ────────────────────

#' Top-k filtered softmax (Fan 2018)
#' @keywords internal
top_k_decoding <- function(x, k = 5L, T = 1) {
  z <- as.numeric(x) / T
  Vlen <- length(z)
  k <- max(1L, min(as.integer(k), Vlen))
  thresh <- sort(z, decreasing = TRUE)[k]
  z_f <- ifelse(z >= thresh, z, -Inf)
  z_f <- z_f - max(z_f)
  e <- exp(z_f); p <- e / sum(e)
  topk_idx <- order(z, decreasing = TRUE)[seq_len(k)]
  list(tensor = p, topk_indices = topk_idx - 1L,
       topk_logits = z[topk_idx], k = k, method = "top-k")
}


# ──────────────────── 10. toppd / top_p_nucleus ─────────────────────

#' Top-p nucleus sampling (Holtzman 2020)
#' @keywords internal
top_p_nucleus <- function(x, p = 0.9, T = 1) {
  if (p <= 0 || p > 1) stop("p must be in (0, 1]")
  z <- as.numeric(x) / T; z <- z - max(z)
  probs <- exp(z); probs <- probs / sum(probs)
  ord <- order(-probs)
  cs <- cumsum(probs[ord])
  cutoff <- max(1L, min(which(cs >= p)[1L], length(probs)))
  keep <- logical(length(probs))
  keep[ord[seq_len(cutoff)]] <- TRUE
  filtered <- ifelse(keep, probs, 0)
  filtered <- filtered / sum(filtered)
  list(tensor = filtered, keep_mask = keep, n_kept = sum(keep),
       p = p, method = "top-p")
}


# ────────────────── 11. rptpn / repetition_penalty ──────────────────

#' Repetition penalty (Keskar 2019)
#' @keywords internal
repetition_penalty <- function(x, generated, alpha = 1.2) {
  z <- as.numeric(x)
  if (alpha == 1) return(list(tensor = z, penalised_idx = integer(0),
                               alpha = alpha, method = "rep-penalty"))
  idx <- unique(as.integer(generated))
  idx <- idx[idx >= 0L & idx < length(z)]
  for (i in idx)
    z[i + 1L] <- if (z[i + 1L] > 0) z[i + 1L] / alpha else z[i + 1L] * alpha
  list(tensor = z, penalised_idx = idx, alpha = alpha,
       method = "rep-penalty")
}


# ─────────────────── 12. pplxm / perplexity_metric ──────────────────

#' Perplexity (Jelinek 1977)
#' @keywords internal
perplexity_metric <- function(x, base = "e") {
  logp <- as.numeric(x)
  if (!length(logp)) stop("Need at least one token log-prob")
  if (identical(base, "2")) logp <- logp * log(2)
  else if (!identical(base, "e")) stop("base must be 'e' or '2'")
  nll <- -mean(logp); ppl <- exp(nll)
  list(value = ppl, nll = nll, n = length(logp),
       method = "perplexity")
}


# ──────────────────── 13. bpblm / bits_per_byte ─────────────────────

#' Bits per byte (Gao 2020)
#' @keywords internal
bits_per_byte <- function(x, n_bytes = NULL) {
  nll <- as.numeric(x)
  if (!length(nll)) stop("Need at least one token NLL")
  total <- sum(nll); nb <- if (is.null(n_bytes)) length(nll) else as.integer(n_bytes)
  if (nb <= 0) stop("n_bytes must be > 0")
  list(value = total / (nb * log(2)),
       nll_nats = total, n_tokens = length(nll), n_bytes = nb,
       method = "BPB")
}


# ─────────────── 14. cslnc / cosine_lr_schedule ─────────────────────

#' Cosine LR schedule with warmup (Loshchilov 2017)
#' @keywords internal
cosine_lr_schedule <- function(x, lr_max = 1e-3, lr_min = 0,
                                total_steps = 1000L, warmup_steps = 0L) {
  if (total_steps <= warmup_steps)
    stop("total_steps must exceed warmup_steps")
  t <- as.numeric(x)
  warm <- t < warmup_steps
  lr <- numeric(length(t))
  lr[warm] <- lr_max * t[warm] / max(1, warmup_steps)
  dec <- pmin(pmax((t - warmup_steps) /
                     (total_steps - warmup_steps), 0), 1)
  lr[!warm] <- lr_min + 0.5 * (lr_max - lr_min) *
    (1 + cos(pi * dec[!warm]))
  list(value = lr[1L], tensor = lr, step = t,
       lr_max = lr_max, lr_min = lr_min,
       total_steps = total_steps, warmup_steps = warmup_steps,
       method = "cosine-LR")
}


# ─────────────────── 15. grdcl / gradient_clipping ──────────────────

#' Global-norm gradient clip (Pascanu 2013)
#' @keywords internal
gradient_clipping <- function(x, max_norm = 1) {
  is_list <- is.list(x)
  cat_vec <- if (is_list) unlist(lapply(x, as.numeric))
             else as.numeric(x)
  total <- sqrt(sum(cat_vec * cat_vec))
  coef <- min(1, max_norm / (total + 1e-12))
  clipped <- if (is_list) lapply(x, function(g) as.numeric(g) * coef)
             else as.numeric(x) * coef
  list(tensor = clipped, clip_coef = coef,
       total_norm = total, max_norm = max_norm,
       method = "global-norm-clip")
}


# ────────────────────── 16. lradw / lr_warmup ───────────────────────

#' Linear LR warmup (Vaswani 2017)
#' @keywords internal
lr_warmup <- function(x, lr_target = 1e-3, warmup_steps = 1000L) {
  if (warmup_steps <= 0) stop("warmup_steps must be > 0")
  t <- as.numeric(x)
  lr <- lr_target * pmin(1, t / warmup_steps)
  list(tensor = lr, value = lr[1L],
       lr_target = lr_target, warmup_steps = warmup_steps,
       step = t, method = "linear-warmup")
}


# ──────────────────── 17. flshA / flash_attention ───────────────────

#' FlashAttention (Dao 2022) — IO-aware tiled softmax
#' @keywords internal
flash_attention <- function(Q, K = NULL, V = NULL, block_size = 32L,
                            mask = NULL) {
  if (is.null(K)) K <- Q
  if (is.null(V)) V <- Q
  Q <- as.matrix(Q); K <- as.matrix(K); V <- as.matrix(V)
  N <- nrow(Q); d <- ncol(Q); M <- nrow(K)
  scale <- 1 / sqrt(d)
  out <- matrix(0, N, d)
  row_max <- rep(-Inf, N)
  row_den <- rep(0, N)
  j <- 1L
  while (j <= M) {
    je <- min(j + block_size - 1L, M)
    Kj <- K[j:je, , drop = FALSE]
    Vj <- V[j:je, , drop = FALSE]
    s <- (Q %*% t(Kj)) * scale
    if (!is.null(mask)) s <- s + as.matrix(mask)[, j:je, drop = FALSE]
    bm <- apply(s, 1L, max)
    new_max <- pmax(row_max, bm)
    alpha <- exp(row_max - new_max)              # length N
    beta <- exp(sweep(s, 1L, new_max, "-"))      # N x k, row-wise
    row_den <- row_den * alpha + rowSums(beta)
    out <- sweep(out, 1L, alpha, "*") + beta %*% Vj
    row_max <- new_max
    j <- je + 1L
  }
  out <- sweep(out, 1L, row_den, "/")
  list(tensor = out, block_size = block_size,
       method = "flash-attention")
}


# ──────────────────── 18. spqkv / sparse_attention ──────────────────

#' Child-2019 sparse attention mask
#' @keywords internal
sparse_attention <- function(x, window = 4L, stride = 8L,
                              n_random = 0L, seed = 0L) {
  N <- if (length(x) == 1L && is.numeric(x)) as.integer(x)
       else if (!is.null(dim(x))) dim(x)[length(dim(x)) - 1L]
       else length(x)
  set.seed(seed)
  M <- matrix(FALSE, N, N)
  for (i in seq_len(N)) {
    lo <- max(1L, i - window); hi <- min(N, i + window)
    M[i, lo:hi] <- TRUE
    M[i, seq.int(1L, N, by = stride)] <- TRUE
    if (n_random > 0L) {
      picks <- sample.int(N, size = min(n_random, N))
      M[i, picks] <- TRUE
    }
  }
  additive <- ifelse(M, 0, -Inf)
  density <- sum(M) / (N * N)
  list(tensor = additive, boolean = M, density = density,
       method = "sparse-attention")
}


# ───────────────── 19. moeml / mixture_of_experts ───────────────────

#' Sparsely-gated MoE (Shazeer 2017)
#' @keywords internal
mixture_of_experts <- function(x, W_gate = NULL, experts = NULL,
                                top_k = 2L) {
  xm <- as.matrix(x)
  B <- nrow(xm); d_in <- ncol(xm)
  if (is.null(W_gate)) {
    n_experts <- 2L; W_gate <- matrix(0, d_in, n_experts)
  }
  n_experts <- ncol(W_gate)
  if (is.null(experts)) {
    experts <- replicate(n_experts,
                         list(W = diag(d_in), b = rep(0, d_in)),
                         simplify = FALSE)
  }
  gate_logits <- xm %*% W_gate
  gate <- t(apply(gate_logits, 1L, function(v) {
    v <- v - max(v); e <- exp(v); e / sum(e)
  }))
  k <- max(1L, min(as.integer(top_k), n_experts))
  topk_idx <- t(apply(gate, 1L, function(g)
    order(-g)[seq_len(k)]))
  sparse <- matrix(0, B, n_experts)
  for (b in seq_len(B))
    sparse[b, topk_idx[b, ]] <- gate[b, topk_idx[b, ]]
  sparse <- sparse / rowSums(sparse)
  expert_outs <- lapply(experts, function(e)
    sweep(xm %*% e$W, 2L, e$b, "+"))
  d_out <- ncol(expert_outs[[1L]])
  y <- matrix(0, B, d_out)
  for (e in seq_len(n_experts))
    y <- y + sweep(expert_outs[[e]], 1L, sparse[, e], "*")
  load <- colSums(sparse) / B
  list(tensor = y, gate = sparse,
       topk_idx = topk_idx - 1L, load = load, method = "MoE")
}


# ───────────────────── 20. rlhfd / rlhf_reward ──────────────────────

#' RLHF linear reward head (Ouyang 2022)
#' @keywords internal
rlhf_reward <- function(x, w = NULL, b = 0) {
  xm <- as.matrix(x)
  d <- ncol(xm)
  if (is.null(w)) w <- rep(1 / d, d)
  if (length(w) != d) stop(sprintf("w must have length %d", d))
  r <- as.numeric(xm %*% w + b)
  list(value = r[1L], tensor = r,
       w = as.numeric(w), b = b, method = "rlhf-reward-head")
}


# ─────────────────────── CANONICAL TESTS ────────────────────────────
# # tknbp
# stopifnot(bpe_tokenizer(c("low","low","lower","newest","newest","newest"),
#                         num_merges=3L)$n_merges == 3L)
# # wdemb
# stopifnot(all.equal(word_embedding(c(0,2), E = diag(4))$tensor,
#                     diag(4)[c(1,3), ]))
# # cslat
# stopifnot(causal_attention_mask(3L)$tensor[1,2] == -Inf)
# # rmsnr
# stopifnot(all.equal(rms_norm(matrix(c(3,4), 1, 2), eps=0)$tensor[1,],
#                     c(3,4) / sqrt(12.5)))
# # tmpsc
# stopifnot(abs(sum(temperature_scaling(c(1,2,3), T=1)$tensor) - 1) < 1e-9)
# # pplxm
# stopifnot(abs(perplexity_metric(c(log(0.5), log(0.5)))$value - 2) < 1e-9)
# # bpblm
# stopifnot(abs(bits_per_byte(rep(log(2), 4), n_bytes=4)$value - 1) < 1e-9)
# # lradw
# stopifnot(abs(lr_warmup(500, lr_target=1, warmup_steps=1000)$value - 0.5) < 1e-9)
# # grdcl
# stopifnot(abs(sqrt(sum(gradient_clipping(c(3,4), max_norm=1)$tensor^2)) - 1) < 1e-9)
# # rlhfd
# stopifnot(abs(rlhf_reward(matrix(c(1,1),1,2), w=c(0.5,0.5))$value - 1) < 1e-9)
