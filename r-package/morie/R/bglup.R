# SPDX-License-Identifier: MIT OR Apache-2.0

#' BayesC-pi spike-and-slab variable selection (short Gibbs)
#'
#' @param x (n x p) marker matrix.
#' @param y Numeric response.
#' @param n_iter Iterations.
#' @param burn Burn-in.
#' @param pi_init Initial inclusion probability.
#' @param seed Seed.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("bglup", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return list(estimate, beta, beta_pip, pi, sigma_b2, sigma2, n_iter, n, p, method).
#' @references Habier-Fernando-Kizilkaya-Garrick (2011); Montesinos Lopez Ch 4.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
bayes_cpi_genomic <- function(x, y, n_iter = 300, burn = 100,
                               pi_init = 0.1, seed = 0,
                               deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("bglup", deterministic_seed)
  } else {
    set.seed(seed)
  }
  X <- as.matrix(x); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  ym <- mean(y); yc <- y - ym
  Xc <- sweep(X, 2, colMeans(X))
  var_y <- if (n > 1) stats::var(yc) else 1
  sigma_b2 <- var_y / max(p, 1); sigma2 <- var_y
  pi_in <- as.numeric(pi_init)
  df_b <- 4; S_b <- max(sigma_b2 * (df_b - 2) / df_b, 1e-6)
  df_e <- 4; S_e <- max(var_y * (df_e - 2) / df_e, 1e-6)
  beta <- rep(0, p); delta <- rep(0L, p)
  r <- yc; xtx_diag <- colSums(Xc^2)
  bsum <- matrix(0, 0, p); psum <- matrix(0, 0, p)
  piv <- numeric(0); sbv <- numeric(0); sv2 <- numeric(0)
  for (it in seq_len(n_iter)) {
    for (j in seq_len(p)) {
      xj <- Xc[, j]
      r_j <- r + xj * beta[j]
      v <- xtx_diag[j] / sigma2 + 1 / sigma_b2
      mn <- sum(xj * r_j) / sigma2 / v
      log_bf <- 0.5 * log(1 / max(sigma_b2 * v, 1e-30)) + 0.5 * v * mn^2
      log_pi <- log(max(pi_in, 1e-30))
      log_1mp <- log(max(1 - pi_in, 1e-30))
      log_p1 <- log_pi + log_bf; log_p0 <- log_1mp
      mx <- max(log_p1, log_p0)
      prob_in <- exp(log_p1 - mx) / (exp(log_p1 - mx) + exp(log_p0 - mx))
      delta[j] <- as.integer(stats::runif(1) < prob_in)
      if (delta[j] == 1L) beta[j] <- stats::rnorm(1, mn, 1 / sqrt(v))
      else beta[j] <- 0
      r <- r_j - xj * beta[j]
    }
    k_in <- sum(delta)
    pi_in <- stats::rbeta(1, 1 + k_in, 1 + p - k_in)
    df_post <- df_b + max(k_in, 1)
    scale_post <- (S_b * df_b + sum(beta[delta == 1L]^2)) / df_post
    sigma_b2 <- max(scale_post * df_post /
                      max(stats::rchisq(1, df_post), 1e-8), 1e-12)
    df_post_e <- df_e + n
    scale_post_e <- (S_e * df_e + sum(r^2)) / df_post_e
    sigma2 <- max(scale_post_e * df_post_e /
                    max(stats::rchisq(1, df_post_e), 1e-8), 1e-12)
    if (it > burn) {
      bsum <- rbind(bsum, beta); psum <- rbind(psum, delta)
      piv <- c(piv, pi_in); sbv <- c(sbv, sigma_b2); sv2 <- c(sv2, sigma2)
    }
  }
  beta_hat <- colMeans(bsum)
  pip <- colMeans(psum)
  list(estimate = mean(abs(beta_hat)), beta = beta_hat, beta_pip = pip,
       pi = mean(piv), sigma_b2 = mean(sbv), sigma2 = mean(sv2),
       intercept = ym, n_iter = length(sv2), n = n, p = p,
       method = "BayesC-pi short Gibbs")
}

# CANONICAL TEST
# set.seed(11); X <- matrix(rnorm(180), 30, 6); b <- c(1,0,0,-1,0,0)
# y <- X %*% b + 0.1*rnorm(30); bayes_cpi_genomic(X, y, seed=11)$beta_pip
