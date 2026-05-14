# SPDX-License-Identifier: MIT OR Apache-2.0

#' Vanilla batch gradient descent for OLS (R parity)
#'
#' theta := theta - lr * (2/n) X' (X theta - y), intercept included.
#' Validates against \code{stats::lm} reference.
#'
#' @param x Numeric matrix / vector of predictors.
#' @param y Numeric response vector.
#' @param lr Learning rate.
#' @param n_iter Max iterations.
#' @param tol L2 step-norm tolerance for early stopping.
#' @return Named list with \code{estimate}, \code{reference_ols},
#'   \code{n_iter}, \code{loss}, \code{n}, \code{method}.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
gradient_descent_vanilla <- function(x, y, lr = 0.01, n_iter = 1000, tol = 1e-8) {
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x); y <- as.numeric(y)
  n <- nrow(x); p <- ncol(x)
  X1 <- cbind(1, x)
  theta <- rep(0.0, p + 1)
  it <- 0L
  for (k in seq_len(n_iter)) {
    grad <- (2 / n) * crossprod(X1, X1 %*% theta - y)
    step <- lr * grad
    theta <- theta - as.numeric(step)
    it <- k
    if (sqrt(sum(step^2)) < tol) break
  }
  loss <- mean((X1 %*% theta - y)^2)
  ref <- stats::lm.fit(X1, y)$coefficients
  list(
    estimate      = as.numeric(theta),
    reference_ols = unname(ref),
    n_iter        = as.integer(it),
    loss          = as.numeric(loss),
    n             = n,
    method        = "Vanilla batch gradient descent (linear regression)"
  )
}
