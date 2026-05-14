# SPDX-License-Identifier: MIT OR Apache-2.0

#' K-fold cross-validation for genomic-prediction accuracy
#'
#' @param x (n x p) predictor matrix.
#' @param y Numeric response.
#' @param K Number of folds.
#' @param lam Ridge penalty within each fold.
#' @param seed Seed.
#' @return list(estimate, r_per_fold, y_hat, mse, mspe, slope, n, K, method).
#' @references Montesinos Lopez Ch 2.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
genomic_cross_validation <- function(x, y, K = 5, lam = 1.0, seed = 0) {
  set.seed(seed)
  X <- as.matrix(x); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  idx <- sample.int(n)
  folds <- split(idx, cut(seq_along(idx), K, labels = FALSE))
  y_hat <- rep(0, n); r_per_fold <- numeric(K)
  for (k in seq_len(K)) {
    test <- folds[[k]]
    train <- setdiff(seq_len(n), test)
    Xtr <- X[train, , drop = FALSE]; ytr <- y[train]
    Xte <- X[test, , drop = FALSE]
    mu <- mean(ytr); x_mu <- colMeans(Xtr)
    Xtr_c <- sweep(Xtr, 2, x_mu)
    beta <- solve(crossprod(Xtr_c) + lam * diag(p),
                  crossprod(Xtr_c, ytr - mu))
    y_hat[test] <- as.numeric(sweep(Xte, 2, x_mu) %*% beta) + mu
    if (length(test) > 1 && stats::sd(y[test]) > 0 && stats::sd(y_hat[test]) > 0)
      r_per_fold[k] <- stats::cor(y[test], y_hat[test])
    else r_per_fold[k] <- NA_real_
  }
  r_pooled <- if (stats::sd(y_hat) > 0) stats::cor(y, y_hat) else NA_real_
  mse <- mean((y - y_hat)^2); mspe <- mse
  slope <- if (stats::var(y_hat) > 0)
    stats::cov(y_hat, y) / stats::var(y) else NA_real_
  list(estimate = r_pooled, r_per_fold = r_per_fold,
       y_hat = y_hat, mse = mse, mspe = mspe, slope = slope,
       n = n, K = K, method = "K-fold cross-validation (ridge)")
}

# CANONICAL TEST
# set.seed(15); X <- matrix(rnorm(200), 50, 4); b <- c(1,-1,0.5,0)
# y <- X %*% b + 0.3*rnorm(50); genomic_cross_validation(X, y, K=5, seed=15)
