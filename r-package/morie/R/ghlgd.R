# SPDX-License-Identifier: GPL-2.0-only

#' Log-spline density estimator (Stone 1990, Ghosal Ch 8).
#'
#' @param x Numeric data vector.
#' @param K Integer polynomial degree (default 5).
#' @param grid Optional numeric evaluation grid.
#' @return Named list with estimate, theta, log_lik, grid, log_density, K, n, method.
#' @importFrom utils head tail
#' @export
ghosal_log_density <- function(x, K = 5, grid = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5) return(list(estimate = NA_real_, n = n,
                          method = "Log-density (n<5)"))
  m <- mean(x); s <- max(sd(x), 1e-6); z <- (x - m) / s
  if (is.null(grid)) gz <- seq(min(z) - 1, max(z) + 1, length.out = 401)
  else gz <- (grid - m) / s
  basis <- function(u) sapply(seq_len(K), function(k) u^k)
  Bx <- basis(z); Bg <- basis(gz)
  neg_ll <- function(theta) {
    eta_x <- Bx %*% theta; eta_g <- Bg %*% theta
    M <- max(eta_g)
    Z <- M + log(sum(diff(gz) * (head(exp(eta_g - M), -1) +
                                  tail(exp(eta_g - M), -1))) / 2)
    -(sum(eta_x) - n * Z) + 1e-4 * sum(theta^2)
  }
  opt <- stats::optim(rep(0, K), neg_ll, method = "BFGS")
  theta <- opt$par
  eta_g <- Bg %*% theta; M <- max(eta_g)
  logZ <- M + log(sum(diff(gz) * (head(exp(eta_g - M), -1) +
                                   tail(exp(eta_g - M), -1))) / 2)
  log_density <- as.numeric(eta_g - logZ - log(s))
  eta0 <- as.numeric(basis(0) %*% theta)
  est <- eta0 - logZ - log(s)
  list(estimate = est, theta = theta,
       log_lik = -(opt$value - 1e-4 * sum(theta^2)),
       grid = gz * s + m, log_density = log_density, K = K, n = n,
       method = "Log-spline density (Stone 1990)")
}
