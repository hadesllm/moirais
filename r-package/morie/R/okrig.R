# SPDX-License-Identifier: MIT OR Apache-2.0
#' Ordinary kriging prediction (exponential / gaussian / spherical).
#'
#' \deqn{\hat Z(s_0) = \lambda^\top Z,
#'   \begin{bmatrix}C & 1 \\ 1^\top & 0\end{bmatrix}
#'   \begin{bmatrix}\lambda \\ \mu\end{bmatrix} =
#'   \begin{bmatrix}c_0 \\ 1\end{bmatrix}}
#'
#' @param x Numeric vector.
#' @param coords Numeric coordinate matrix (n by d).
#' @param target Numeric (m by d) or (d,) target coords.
#' @param model "exponential", "gaussian", or "spherical".
#' @param nugget,sill,range_ Variogram parameters.
#' @return Named list: estimate, se, n, method.
#' @references Schabenberger & Gotway (2005), Ch 4.
#' @export
okrig <- function(x, coords, target, model = "exponential",
                  nugget = 0, sill = 1, range_ = 1) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  target <- if (is.matrix(target)) target else
    matrix(as.numeric(unlist(target)), ncol = ncol(coords))
  if (nrow(coords) != n) stop("coords rows must match length(x)")
  if (ncol(target) != ncol(coords)) stop("target dim mismatch")
  c0 <- nugget; c1 <- sill - nugget; a <- range_
  cov_fn <- function(h) {
    switch(model,
           exponential = c1 * exp(-h / a) + ifelse(h == 0, c0, 0),
           gaussian    = c1 * exp(-(h ^ 2) / (a ^ 2)) + ifelse(h == 0, c0, 0),
           spherical   = ifelse(h <= a,
                                c1 * (1 - 1.5 * h / a + 0.5 * (h / a) ^ 3),
                                0) + ifelse(h == 0, c0, 0),
           stop("unknown model"))
  }
  Dnn <- as.matrix(stats::dist(coords))
  C <- cov_fn(Dnn)
  A <- matrix(0, n + 1, n + 1)
  A[1:n, 1:n] <- C; A[1:n, n + 1] <- 1; A[n + 1, 1:n] <- 1
  total_var <- c0 + c1
  m <- nrow(target)
  ests <- numeric(m); ses <- numeric(m)
  for (k in seq_len(m)) {
    d0 <- sqrt(colSums((t(coords) - target[k, ]) ^ 2))
    c_vec <- cov_fn(d0)
    rhs <- c(c_vec, 1)
    sol <- tryCatch(solve(A, rhs),
                    error = function(e) qr.solve(A, rhs))
    lam <- sol[1:n]; mu <- sol[n + 1]
    ests[k] <- sum(lam * x)
    var_pred <- max(total_var - sum(lam * c_vec) - mu, 0)
    ses[k] <- sqrt(var_pred)
  }
  list(estimate = if (m == 1) ests[1] else ests,
       se = if (m == 1) ses[1] else ses,
       n = n, method = sprintf("Ordinary kriging (%s)", model))
}

# CANONICAL TEST
# okrig(c(1,2,3,4,5), matrix(0:4, ncol=1), matrix(2.5, 1, 1),
#       "exponential", 0, 1, 2)$estimate  # ~ 3.5

#' @rdname okrig
#' @keywords internal
#' @export
ordinary_kriging <- okrig
