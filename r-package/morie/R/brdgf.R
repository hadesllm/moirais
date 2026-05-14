# SPDX-License-Identifier: MIT OR Apache-2.0

#' BayesA via short Gibbs sampler (Meuwissen-Hayes-Goddard 2001)
#'
#' Per-marker variance with scaled inverse chi-squared prior.
#'
#' @param x (n x p) marker matrix.
#' @param y Numeric response.
#' @param n_iter Iterations.
#' @param burn Burn-in.
#' @param df0 Prior df (default 4).
#' @param S0 Prior scale (default anchors to var(y)/p).
#' @param seed Seed.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("brdgf", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return list(estimate, beta, beta_se, sigma_j2, sigma2, n_iter, n, p, method).
#' @references Meuwissen-Hayes-Goddard (2001) Genetics 157:1819.
#' @export
bayes_ridge_gibbs <- function(x, y, n_iter = 200, burn = 50,
                               df0 = 4, S0 = NULL, seed = 0,
                               deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("brdgf", deterministic_seed)
  } else {
    set.seed(seed)
  }
  X <- as.matrix(x); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  ym <- mean(y); yc <- y - ym
  Xc <- sweep(X, 2, colMeans(X))
  var_y <- if (n > 1) stats::var(yc) else 1
  if (is.null(S0)) S0 <- max((var_y / max(p, 1)) * (df0 - 2) / df0, 1e-6)
  beta <- rep(0, p)
  sigma_j2 <- rep(var_y / max(p, 1), p)
  sigma2 <- var_y
  xtx_diag <- colSums(Xc^2)
  resid <- yc - as.numeric(Xc %*% beta)
  bsum <- matrix(0, 0, p); sj_sum <- matrix(0, 0, p); ssum <- numeric(0)
  for (it in seq_len(n_iter)) {
    for (j in seq_len(p)) {
      xj <- Xc[, j]
      r_j <- resid + xj * beta[j]
      v <- xtx_diag[j] / sigma2 + 1 / sigma_j2[j]
      mn <- (sum(xj * r_j)) / sigma2 / v
      beta[j] <- stats::rnorm(1, mn, 1 / sqrt(v))
      resid <- r_j - xj * beta[j]
    }
    df_post <- df0 + 1
    scale_post <- (S0 * df0 + beta^2) / df_post
    chi2 <- stats::rchisq(p, df_post)
    sigma_j2 <- pmax(scale_post * df_post / pmax(chi2, 1e-8), 1e-12)
    df_e <- 4; Se <- var_y * (df_e - 2) / df_e
    df_post_e <- n + df_e
    scale_post_e <- (sum(resid^2) + df_e * Se) / df_post_e
    sigma2 <- max(scale_post_e * df_post_e /
                    max(stats::rchisq(1, df_post_e), 1e-8), 1e-12)
    if (it > burn) {
      bsum <- rbind(bsum, beta); sj_sum <- rbind(sj_sum, sigma_j2)
      ssum <- c(ssum, sigma2)
    }
  }
  beta_hat <- colMeans(bsum)
  beta_se <- if (nrow(bsum) > 1) apply(bsum, 2, stats::sd) else rep(0, p)
  sigma_j2_hat <- colMeans(sj_sum)
  list(estimate = mean(abs(beta_hat)), beta = beta_hat, beta_se = beta_se,
       se = mean(beta_se), sigma_j2 = sigma_j2_hat,
       sigma2 = mean(ssum), intercept = ym,
       n_iter = length(ssum), n = n, p = p,
       method = "BayesA short Gibbs (Meuwissen-Hayes-Goddard)")
}

# CANONICAL TEST
# set.seed(4); X <- matrix(rnorm(100), 20, 5); b <- c(1,-1,0.5,0,0)
# y <- X %*% b + 0.2*rnorm(20); bayes_ridge_gibbs(X, y, seed=4)$beta
