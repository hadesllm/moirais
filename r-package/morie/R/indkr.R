# SPDX-License-Identifier: MIT OR Apache-2.0
#' Indicator kriging for exceedance probability.
#'
#' Encode I_i = 1(x_i <= threshold) and ordinary-krige the indicator
#' field at each target s0 to obtain P_hat(Z(s0) <= threshold).
#'
#' @param x Numeric vector.
#' @param coords Coord matrix.
#' @param threshold Numeric scalar.
#' @param target Optional target coords; defaults to sample coords.
#' @param nugget,sill,range_ Indicator-covariance parameters.
#' @return Named list: estimate, threshold, n, method.
#' @references Journel (1983); Schabenberger & Gotway (2005), Ch 4.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
indkr <- function(x, coords, threshold, target = NULL,
                  nugget = 0, sill = 0.25, range_ = 1) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  if (is.null(target)) target <- coords
  target <- if (is.matrix(target)) target else
    matrix(as.numeric(unlist(target)), ncol = ncol(coords))
  if (nrow(coords) != n) stop("coords rows must match length(x)")
  if (ncol(target) != ncol(coords)) stop("target/coords dim mismatch")
  I_obs <- as.numeric(x <= threshold)
  c0 <- nugget; c1 <- sill - nugget; a <- range_
  D <- as.matrix(stats::dist(coords))
  cov_fn <- function(h) c1 * exp(-h / a) + ifelse(h == 0, c0, 0)
  C <- cov_fn(D)
  A <- matrix(0, n + 1, n + 1)
  A[1:n, 1:n] <- C; A[1:n, n + 1] <- 1; A[n + 1, 1:n] <- 1
  m <- nrow(target)
  probs <- numeric(m)
  for (k in seq_len(m)) {
    d0 <- sqrt(colSums((t(coords) - target[k, ]) ^ 2))
    c_vec <- cov_fn(d0)
    rhs <- c(c_vec, 1)
    sol <- tryCatch(solve(A, rhs),
                    error = function(e) qr.solve(A, rhs))
    lam <- sol[1:n]
    probs[k] <- min(max(sum(lam * I_obs), 0), 1)
  }
  list(estimate = if (m == 1) probs[1] else probs,
       threshold = threshold, n = n,
       method = "Indicator kriging (ordinary, exp. cov)")
}

#' @rdname indkr
#' @keywords internal
#' @export
indicator_kriging <- indkr
