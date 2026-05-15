# SPDX-License-Identifier: AGPL-3.0-or-later

#' Sparsely-gated MoE (Shazeer 2017)
#'
#' @param x Input matrix (B by d_in).
#' @param W_gate Optional gating weights (d_in by n_experts).
#' @param experts Optional list of expert weight/bias pairs.
#' @param top_k Integer experts kept per token (default 2).
#' @return Named list with tensor, gate, topk_idx, load, method.
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
