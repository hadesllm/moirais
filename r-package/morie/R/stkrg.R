# SPDX-License-Identifier: MIT OR Apache-2.0
#' Spatiotemporal ordinary kriging with separable exponential covariance.
#'
#' C((h, u)) = (sill - nugget) * exp(-h/range_s) * exp(-|u|/range_t)
#'              + nugget if (h, u) == (0, 0).
#'
#' @param x Numeric vector.
#' @param coords Spatial coord matrix.
#' @param times Numeric times.
#' @param target list(s0 = (m by d) matrix, t0 = numeric m).
#' @param sill,nugget,range_s,range_t Variogram parameters.
#' @return Named list: estimate, se, n, method.
#' @references Schabenberger & Gotway (2005), Ch 8.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
stkrg <- function(x, coords, times, target,
                  sill = 1, nugget = 0, range_s = 1, range_t = 1) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  t <- as.numeric(times)
  if (nrow(coords) != n || length(t) != n) stop("shape mismatch")
  s0 <- as.matrix(target$s0); t0 <- as.numeric(target$t0)
  if (ncol(s0) != ncol(coords)) stop("target coords dim mismatch")
  if (nrow(s0) != length(t0)) stop("target s0 and t0 must align")
  c0 <- nugget; c1 <- sill - nugget
  Dnn <- as.matrix(stats::dist(coords))
  Tnn <- abs(outer(t, t, "-"))
  Cmat <- c1 * exp(-Dnn / range_s) * exp(-Tnn / range_t) + c0 * diag(n)
  A <- matrix(0, n + 1, n + 1)
  A[1:n, 1:n] <- Cmat; A[1:n, n + 1] <- 1; A[n + 1, 1:n] <- 1
  var_total <- sill
  m <- nrow(s0)
  ests <- numeric(m); ses <- numeric(m)
  for (k in seq_len(m)) {
    d0s <- sqrt(colSums((t(coords) - s0[k, ]) ^ 2))
    d0t <- abs(t0[k] - t)
    c_vec <- c1 * exp(-d0s / range_s) * exp(-d0t / range_t)
    rhs <- c(c_vec, 1)
    sol <- tryCatch(solve(A, rhs),
                    error = function(e) qr.solve(A, rhs))
    lam <- sol[1:n]; mu <- sol[n + 1]
    ests[k] <- sum(lam * x)
    ses[k] <- sqrt(max(var_total - sum(lam * c_vec) - mu, 0))
  }
  list(estimate = if (m == 1) ests[1] else ests,
       se = if (m == 1) ses[1] else ses, n = n,
       method = "Spatiotemporal ordinary kriging (separable exponential)")
}

# CANONICAL TEST -- predict at an observed site, should reproduce z

#' @rdname stkrg
#' @keywords internal
#' @export
spatiotemporal_kriging <- stkrg
