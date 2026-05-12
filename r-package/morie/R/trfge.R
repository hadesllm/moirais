# SPDX-License-Identifier: GPL-2.0-only

#' Transformer (1-head self-attention) genomic predictor (base R)
#'
#' Random fixed projections + ridge head on the mean-pooled context vector.
#'
#' @param x Optional fixed-effect features.
#' @param y Numeric response.
#' @param markers (n x L) marker sequence.
#' @param d_model Model dimension.
#' @param lam Ridge regulariser for the linear head.
#' @param seed Seed.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("trfge", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return list(estimate, y_hat, beta, attention, context, se, n, method).
#' @references Vaswani et al. (2017). Montesinos Lopez Ch 15.
#' @export
transformer_genomic <- function(x, y, markers, d_model = 8, lam = 1, seed = 0,
                                 deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("trfge", deterministic_seed)
  } else {
    set.seed(seed)
  }
  y <- as.numeric(y); n <- length(y)
  M <- as.matrix(markers); L <- ncol(M)
  M_mu <- colMeans(M); M_sd <- apply(M, 2, stats::sd); M_sd[M_sd == 0] <- 1
  Ms <- sweep(sweep(M, 2, M_mu), 2, M_sd, "/")
  sc <- 1 / sqrt(d_model)
  W_emb <- matrix(stats::rnorm(d_model, 0, sc), 1, d_model)
  W_Q <- matrix(stats::rnorm(d_model^2, 0, sc), d_model, d_model)
  W_K <- matrix(stats::rnorm(d_model^2, 0, sc), d_model, d_model)
  W_V <- matrix(stats::rnorm(d_model^2, 0, sc), d_model, d_model)
  pos <- seq_len(L) - 1; dim_idx <- seq_len(d_model) - 1
  div <- 10000^((2 * (dim_idx %/% 2)) / d_model)
  pe <- matrix(0, L, d_model)
  for (i in seq_len(d_model)) {
    if (i %% 2 == 1) pe[, i] <- sin(pos / div[i])
    else pe[, i] <- cos(pos / div[i])
  }
  softmax_row <- function(S) {
    Sx <- S - apply(S, 1, max)
    e <- exp(Sx); sweep(e, 1, rowSums(e), "/")
  }
  context <- matrix(0, n, d_model)
  attention <- array(0, dim = c(n, L, L))
  for (i in seq_len(n)) {
    E <- matrix(Ms[i, ], L, 1) %*% W_emb + pe
    Q <- E %*% W_Q; K <- E %*% W_K; V <- E %*% W_V
    scores <- Q %*% t(K) / sqrt(d_model)
    A <- softmax_row(scores)
    attention[i, , ] <- A
    context[i, ] <- colMeans(A %*% V)
  }
  feats <- cbind(1, context)
  if (!is.null(x)) {
    xa <- as.matrix(x); if (ncol(xa) >= 1 && length(xa) > 0)
      feats <- cbind(feats, xa)
  }
  Amat <- crossprod(feats) + lam * diag(ncol(feats))
  Amat[1, 1] <- Amat[1, 1] - lam
  beta <- as.numeric(solve(Amat, crossprod(feats, y)))
  y_hat <- as.numeric(feats %*% beta)
  resid <- y - y_hat
  list(estimate = mean(y_hat), y_hat = y_hat, beta = beta,
       attention = attention, context = context,
       se = sqrt(mean(resid^2)), n = n,
       method = "Transformer 1-head random-projection + ridge head")
}

# CANONICAL TEST
# set.seed(9); M <- matrix(rnorm(72), 12, 6); y <- M[,3] + 0.2*rnorm(12)
# transformer_genomic(rep(0,12), y, M, seed=9)
