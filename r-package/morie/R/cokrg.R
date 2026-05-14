# SPDX-License-Identifier: MIT OR Apache-2.0
#' Simple cokriging for co-located bivariate spatial prediction.
#'
#' \deqn{\hat Z_1(s_0) = \lambda^\top Z_1 + \mu^\top Z_2}, system
#' \deqn{[C_{pp} \; C_{ps}; C_{ps}^\top \; C_{ss}] [\lambda; \mu] = [c_{0p}; c_{0s}]}.
#'
#' @param x Primary variable (n,).
#' @param y Secondary variable (n,).
#' @param coords Co-located coord matrix.
#' @param target Target coords (m by d) or (d,).
#' @param sill_p,range_p Primary auto-covariance parameters.
#' @param sill_s,range_s Secondary auto-covariance parameters.
#' @param cross_sill,cross_range Cross-covariance parameters.
#' @param nugget Nugget.
#' @return Named list: estimate, se, n, method.
#' @references Schabenberger & Gotway (2005), Ch 4.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
cokrg <- function(x, y, coords, target,
                  sill_p = 1, range_p = 1,
                  sill_s = 1, range_s = 1,
                  cross_sill = 0.5, cross_range = 1,
                  nugget = 0) {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  target <- if (is.matrix(target)) target else
    matrix(as.numeric(unlist(target)), ncol = ncol(coords))
  if (length(y) != n || nrow(coords) != n)
    stop("x, y, and coords must have matching n")
  if (ncol(target) != ncol(coords)) stop("target/coords dim mismatch")
  D <- as.matrix(stats::dist(coords))
  cov_exp <- function(D_, c0, c1, a) c1 * exp(-D_ / a) + ifelse(D_ == 0, c0, 0)
  Cpp <- cov_exp(D, nugget, sill_p - nugget, range_p)
  Css <- cov_exp(D, nugget, sill_s - nugget, range_s)
  Cps <- cross_sill * exp(-D / cross_range)
  C <- rbind(cbind(Cpp, Cps), cbind(t(Cps), Css))
  z <- c(x, y)
  var0 <- sill_p
  m <- nrow(target)
  ests <- numeric(m); ses <- numeric(m)
  for (k in seq_len(m)) {
    d0 <- sqrt(colSums((t(coords) - target[k, ]) ^ 2))
    c0p <- cov_exp(d0, nugget, sill_p - nugget, range_p)
    c0s <- cross_sill * exp(-d0 / cross_range)
    c_vec <- c(c0p, c0s)
    w <- tryCatch(solve(C, c_vec),
                  error = function(e) qr.solve(C, c_vec))
    ests[k] <- sum(w * z)
    ses[k] <- sqrt(max(var0 - sum(w * c_vec), 0))
  }
  list(estimate = if (m == 1) ests[1] else ests,
       se = if (m == 1) ses[1] else ses, n = n,
       method = "Simple cokriging (linear coregionalization, exp. cov)")
}

#' @rdname cokrg
#' @keywords internal
#' @export
cokriging <- cokrg
