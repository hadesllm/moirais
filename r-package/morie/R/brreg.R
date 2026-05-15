# SPDX-License-Identifier: AGPL-3.0-or-later

#' Bayesian ridge regression (RR-BLUP closed form)
#'
#' beta_hat = solve(X'X + lambda*I) %*% X'y
#'
#' @param x (n x p) marker matrix.
#' @param y Numeric response.
#' @param lam Optional ridge parameter; default Endelman rrBLUP-style.
#' @return list(estimate, beta, intercept, se, beta_se, lam, n, p, method).
#' @references Montesinos Lopez Ch 4.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
bayesian_ridge_regression <- function(x, y, lam = NULL) {
  X <- as.matrix(x); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  ym <- mean(y); yc <- y - ym
  xm <- colMeans(X); Xc <- sweep(X, 2, xm)
  if (is.null(lam)) {
    var_y <- if (n > 1) stats::var(yc) else 1
    h2 <- 0.5
    var_e <- (1 - h2) * var_y
    sig_b2 <- (h2 * var_y) / max(p, 1)
    lam <- if (sig_b2 > 0) var_e / sig_b2 else 1
  }
  XtX <- crossprod(Xc)
  A <- XtX + lam * diag(p)
  beta <- as.numeric(solve(A, crossprod(Xc, yc)))
  y_hat <- as.numeric(Xc %*% beta) + ym
  resid <- y - y_hat
  sigma2 <- sum(resid^2) / max(n - 1, 1)
  cov_beta <- sigma2 * solve(A)
  beta_se <- sqrt(pmax(diag(cov_beta), 0))
  list(estimate = mean(abs(beta)), beta = beta, intercept = ym,
       se = mean(beta_se), beta_se = beta_se, lam = lam,
       n = n, p = p, method = "Bayesian ridge (closed-form posterior mode)")
}

# CANONICAL TEST
# set.seed(2); X <- matrix(rnorm(100), 20, 5); b <- c(1,-1,0.5,0,0)
# y <- X %*% b + 0.1*rnorm(20); bayesian_ridge_regression(X, y)$beta
