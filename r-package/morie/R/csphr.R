# SPDX-License-Identifier: MIT OR Apache-2.0

#' Cutting plane (linear discriminant; Armstrong Ch 3)
#'
#' Pooled-LDA cutting hyperplane w'x = c for a binary vote vector;
#' classifies legislators by Mahalanobis-style projection onto the
#' between-class direction.
#'
#' @param x Ideal-point matrix (n by p).
#' @param votes Integer 0/1 vote vector.
#' @return Named list with `w`, `c`, `midpoint`, `correct_class`, `n`,
#'   `p`, `method`.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
csphr <- function(x, votes = NULL) {
  X <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1L)
  n <- nrow(X); p <- ncol(X)
  if (is.null(votes) || n == 0L)
    return(list(w = rep(0, p), c = NA_real_, midpoint = rep(NA_real_, p),
                correct_class = 0L, n = n, p = p,
                method = "cutting_plane_sphere"))
  y <- as.integer(votes)
  Xy <- X[y == 1, , drop = FALSE]; Xn <- X[y == 0, , drop = FALSE]
  if (nrow(Xy) == 0L || nrow(Xn) == 0L) {
    cc <- max(sum(y == 1L), sum(y == 0L))
    return(list(w = rep(0, p), c = NA_real_, midpoint = rep(NA_real_, p),
                correct_class = as.integer(cc), n = n, p = p,
                method = "cutting_plane_sphere"))
  }
  mu_y <- colMeans(Xy); mu_n <- colMeans(Xn)
  S_y <- if (nrow(Xy) > 1L) stats::var(Xy) else matrix(0, p, p)
  S_n <- if (nrow(Xn) > 1L) stats::var(Xn) else matrix(0, p, p)
  S <- ((nrow(Xy) - 1) * S_y + (nrow(Xn) - 1) * S_n) / max(n - 2L, 1L)
  S <- S + 1e-9 * diag(p)
  w <- tryCatch(solve(S, mu_y - mu_n), error = function(e) mu_y - mu_n)
  midpoint <- (mu_y + mu_n) / 2
  c_int <- as.numeric(w %*% midpoint)
  pred <- as.integer(X %*% w > c_int)
  cc <- sum(pred == y)
  if (cc < n - cc) { w <- -w; c_int <- -c_int; cc <- n - cc }
  list(w = w, c = c_int, midpoint = midpoint, correct_class = as.integer(cc),
       n = n, p = p, method = "cutting_plane_sphere")
}

#' @keywords internal
#' @rdname csphr
#' @export
cutting_plane_sphere <- csphr
