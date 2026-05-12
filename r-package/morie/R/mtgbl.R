# SPDX-License-Identifier: GPL-2.0-only

#' Multi-trait GBLUP via vec-stacked mixed-model equations
#'
#' @param x Fixed-effect design (vector or matrix).
#' @param y Multi-trait response (n x t).
#' @param markers Genotype matrix (n x m).
#' @param Sigma_g Optional t x t genetic covariance.
#' @param Sigma_e Optional t x t residual covariance.
#' @return list(estimate, G_hat, B_hat, Sigma_g, Sigma_e, n, t, method).
#' @references Montesinos Lopez Ch 10.
#' @export
multi_trait_gblup <- function(x, y, markers, Sigma_g = NULL, Sigma_e = NULL) {
  Y <- as.matrix(y); n <- nrow(Y); t <- ncol(Y)
  M <- as.matrix(markers)
  G_mat <- grm_vanraden(M, method = 1)$estimate + 1e-6 * diag(n)
  cand <- if (is.null(x) || (is.numeric(x) && length(x) == 0)) {
    matrix(1, n, 1)
  } else cbind(1, as.matrix(x))
  qrx <- qr(cand); X <- cand[, qrx$pivot[seq_len(qrx$rank)], drop = FALSE]
  if (is.null(Sigma_g) || is.null(Sigma_e)) {
    h2 <- 0.5
    S_y <- if (t > 1) stats::cov(Y) else matrix(stats::var(as.numeric(Y)), 1, 1)
    Sigma_g <- h2 * S_y; Sigma_e <- (1 - h2) * S_y
  }
  Sigma_g <- as.matrix(Sigma_g); Sigma_e <- as.matrix(Sigma_e)
  B <- stats::lsfit(X, Y, intercept = FALSE)$coefficients
  if (!is.matrix(B)) B <- matrix(B, ncol = t)
  R <- Y - X %*% B
  V <- kronecker(Sigma_g, G_mat) + kronecker(Sigma_e, diag(n))
  vec_R <- as.numeric(R)
  SG <- kronecker(Sigma_g, G_mat)
  g_vec <- SG %*% solve(V, vec_R)
  G_hat <- matrix(g_vec, n, t)
  list(estimate = mean(G_hat), G_hat = G_hat, B_hat = B,
       Sigma_g = Sigma_g, Sigma_e = Sigma_e,
       n = n, t = t, method = "Multi-trait GBLUP (vec-stacked MME)")
}

# CANONICAL TEST
# set.seed(5); M <- matrix(sample(0:2, 48, TRUE), 6, 8)
# Y <- matrix(rnorm(12), 6, 2); multi_trait_gblup(rep(0,6), Y, M)
