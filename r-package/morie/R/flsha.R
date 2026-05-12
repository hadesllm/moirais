# SPDX-License-Identifier: GPL-2.0-only

#' FlashAttention (Dao 2022) IO-aware tiled softmax
#'
#' @param Q Query matrix (seq_q by d).
#' @param K Optional key matrix; defaults to Q.
#' @param V Optional value matrix; defaults to Q.
#' @param block_size Integer tile size (default 32).
#' @param mask Optional additive attention mask.
#' @return Named list with tensor, block_size, method.
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
