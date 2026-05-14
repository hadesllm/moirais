# SPDX-License-Identifier: MIT OR Apache-2.0
#' Non-stationary covariance estimation (moving-window kernel).
#'
#' C(s_i, s_j) = sigma(s_i) sigma(s_j) rho(s_i, s_j) with kernel-weighted
#' local moments and standardised residual correlations.
#'
#' @param x Numeric vector.
#' @param coords Coord matrix.
#' @param bandwidth Numeric bandwidth (default: median pairwise distance).
#' @return Named list: estimate (sigma_local, C_matrix, bandwidth),
#'   n, method.
#' @references Sampson & Guttorp (1992); Schabenberger & Gotway (2005), Ch 8.
#' @export
nstat <- function(x, coords, bandwidth = NULL) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  if (nrow(coords) != n) stop("coords rows must match length(x)")
  D <- as.matrix(stats::dist(coords))
  if (is.null(bandwidth))
    bandwidth <- if (n > 1) stats::median(D[D > 0]) else 1
  if (bandwidth <= 0) bandwidth <- 1
  K <- exp(-0.5 * (D / bandwidth) ^ 2)
  wsum <- rowSums(K)
  mu_local <- as.numeric((K %*% x) / pmax(wsum, 1))
  dev <- x - mu_local
  var_local <- as.numeric((K %*% (dev ^ 2)) / pmax(wsum, 1))
  sigma_local <- sqrt(pmax(var_local, 1e-12))
  eps <- dev / sigma_local
  rho <- K * outer(eps, eps) / (sqrt(outer(wsum, wsum)) + 1e-12)
  Cmat <- outer(sigma_local, sigma_local) * rho
  list(estimate = list(sigma_local = sigma_local, C_matrix = Cmat,
                       bandwidth = bandwidth),
       n = n, method = "Non-stationary covariance (moving-window kernel)")
}

#' @rdname nstat
#' @keywords internal
#' @export
nonstationary_covariance <- nstat
