# SPDX-License-Identifier: AGPL-3.0-or-later

#' Bayesian LASSO (Park & Casella 2008 short Gibbs)
#'
#' @param x (n x p) marker matrix.
#' @param y Numeric response.
#' @param n_iter Total iterations (default 200).
#' @param burn Burn-in (default 50).
#' @param lam Optional fixed lambda (else empirical-Bayes updated).
#' @param seed Random seed.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("blasf", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return list(estimate, beta, intercept, se, beta_se, lam, sigma2, n_iter, n, p, method).
#' @references Park & Casella (2008) JASA 103:681. Montesinos Lopez Ch 4.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
bayesian_lasso_full <- function(x, y, n_iter = 200, burn = 50,
                                 lam = NULL, seed = 0,
                                 deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("blasf", deterministic_seed)
  } else {
    set.seed(seed)
  }
  X <- as.matrix(x); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  ym <- mean(y); yc <- y - ym
  Xc <- sweep(X, 2, colMeans(X))
  beta <- rep(0, p)
  sigma2 <- if (n > 1) stats::var(yc) else 1
  tau2 <- rep(1, p)
  lam_val <- if (is.null(lam)) 1 else as.numeric(lam)
  XtX <- crossprod(Xc); Xty <- crossprod(Xc, yc)
  bsum <- matrix(0, 0, p); ssum <- numeric(0); lsum <- numeric(0)
  for (it in seq_len(n_iter)) {
    Dinv <- diag(1 / tau2)
    A <- XtX + Dinv
    Aj <- A + diag(1e-8, p)
    L <- chol(Aj)
    mu <- backsolve(L, forwardsolve(t(L), Xty))
    z <- stats::rnorm(p)
    v <- backsolve(L, z) * sqrt(sigma2)
    beta <- as.numeric(mu + v)
    beta_safe <- ifelse(abs(beta) < 1e-8, 1e-8, beta)
    mu_pr <- sqrt(lam_val^2 * sigma2 / beta_safe^2)
    lam_pr <- lam_val^2
    u <- stats::rchisq(p, 1)
    y_ig <- mu_pr + (mu_pr^2 * u) / (2 * lam_pr) -
      (mu_pr / (2 * lam_pr)) *
      sqrt(4 * mu_pr * lam_pr * u + mu_pr^2 * u^2)
    z2 <- stats::runif(p)
    x_ig <- ifelse(z2 <= mu_pr / (mu_pr + y_ig), y_ig, mu_pr^2 / y_ig)
    x_ig <- pmax(x_ig, 1e-8)
    tau2 <- 1 / x_ig
    resid <- yc - as.numeric(Xc %*% beta)
    shape <- (n - 1 + p) / 2
    scale <- 0.5 * (sum(resid^2) + sum(beta^2 / tau2))
    sigma2 <- scale / stats::rgamma(1, shape, 1)
    if (is.null(lam)) {
      shape_l <- p + 1
      rate_l <- 0.5 * sum(tau2) + 0.1
      lam2 <- stats::rgamma(1, shape_l, rate = rate_l)
      lam_val <- sqrt(max(lam2, 1e-8))
    }
    if (it > burn) {
      bsum <- rbind(bsum, beta); ssum <- c(ssum, sigma2); lsum <- c(lsum, lam_val)
    }
  }
  beta_hat <- colMeans(bsum)
  beta_se <- if (nrow(bsum) > 1) apply(bsum, 2, stats::sd) else rep(0, p)
  list(estimate = mean(abs(beta_hat)), beta = beta_hat, intercept = ym,
       se = mean(beta_se), beta_se = beta_se,
       lam = mean(lsum), sigma2 = mean(ssum),
       n_iter = length(ssum), n = n, p = p,
       method = "Bayesian LASSO (Park-Casella short Gibbs)")
}

# CANONICAL TEST
# set.seed(3); X <- matrix(rnorm(100), 20, 5); b <- c(1,-1,0,0,0)
# y <- X %*% b + 0.2*rnorm(20); bayesian_lasso_full(X, y, seed=3)$beta
