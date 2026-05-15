# SPDX-License-Identifier: AGPL-3.0-or-later

#' Grouped-query attention (Ainslie 2023)
#'
#' @param Q Query tensor or matrix.
#' @param K Optional key tensor; defaults to Q.
#' @param V Optional value tensor; defaults to Q.
#' @param n_heads Integer total query heads (default 8).
#' @param n_kv_heads Integer KV head groups (default 2; must divide n_heads).
#' @return Named list with tensor, attn, n_heads, n_kv_heads, group_size, method.
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
