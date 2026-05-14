# SPDX-License-Identifier: MIT OR Apache-2.0

#' Elastic-net regression via coordinate descent (base R)
#'
#' Uses glmnet if available; otherwise the base-R coordinate-descent
#' fallback. Both solve:
#'
#'   min 1/(2n) ||y - X beta||^2 + lam (alpha ||beta||_1 + (1-alpha)/2 ||beta||_2^2).
#'
#' @param x (n x p) predictor matrix.
#' @param y Numeric response.
#' @param alpha 0 (ridge) to 1 (LASSO).
#' @param lam Penalty strength.
#' @param max_iter,tol Convergence controls.
#' @return list(estimate, beta, intercept, se, alpha, lam, n_iter, n, p, method).
#' @references Friedman, Hastie & Tibshirani (2010); Montesinos Lopez Ch 6.
#' @export
penalized_regression <- function(x, y, alpha = 0.5, lam = 1.0,
                                  max_iter = 1000, tol = 1e-6) {
  X <- as.matrix(x); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  use_glmnet <- requireNamespace("glmnet", quietly = TRUE)
  if (use_glmnet) {
    fit <- glmnet::glmnet(X, y, alpha = alpha, lambda = lam,
                           standardize = TRUE, intercept = TRUE)
    beta <- as.numeric(fit$beta[, 1])
    intercept <- as.numeric(fit$a0)
    y_hat <- as.numeric(X %*% beta) + intercept
    resid <- y - y_hat
    se <- sqrt(sum(resid^2) / max(n - p, 1))
    return(list(estimate = mean(abs(beta)), beta = beta, intercept = intercept,
                y_hat = y_hat, se = se, alpha = alpha, lam = lam,
                n_iter = NA_integer_, n = n, p = p,
                method = "glmnet elastic-net"))
  }
  ym <- mean(y); yc <- y - ym
  xm <- colMeans(X); xs <- apply(X, 2, stats::sd); xs[xs == 0] <- 1
  Xs <- sweep(sweep(X, 2, xm), 2, xs, "/")
  beta <- rep(0, p)
  xtx_diag <- colSums(Xs^2) / n
  r <- yc - as.numeric(Xs %*% beta)
  soft <- lam * alpha; ridge_t <- lam * (1 - alpha)
  n_iter_done <- max_iter
  for (it in seq_len(max_iter)) {
    max_change <- 0
    for (j in seq_len(p)) {
      r_j <- r + Xs[, j] * beta[j]
      z <- sum(Xs[, j] * r_j) / n
      if (z > soft) new <- (z - soft) / (xtx_diag[j] + ridge_t)
      else if (z < -soft) new <- (z + soft) / (xtx_diag[j] + ridge_t)
      else new <- 0
      change <- new - beta[j]
      if (abs(change) > max_change) max_change <- abs(change)
      beta[j] <- new
      r <- r_j - Xs[, j] * new
    }
    if (max_change < tol) { n_iter_done <- it; break }
  }
  beta_orig <- beta / xs
  intercept <- ym - sum(xm * beta_orig)
  y_hat <- as.numeric(X %*% beta_orig) + intercept
  resid <- y - y_hat
  se <- sqrt(sum(resid^2) / max(n - p, 1))
  list(estimate = mean(abs(beta_orig)), beta = beta_orig,
       intercept = intercept, y_hat = y_hat, se = se,
       alpha = alpha, lam = lam, n_iter = n_iter_done,
       n = n, p = p, method = "Elastic-net coord descent (base R)")
}

# CANONICAL TEST
# set.seed(10); X <- matrix(rnorm(120), 30, 4); b <- c(1,0,-1,0)
# y <- X %*% b + 0.1*rnorm(30); penalized_regression(X, y, alpha=1, lam=0.05)$beta
