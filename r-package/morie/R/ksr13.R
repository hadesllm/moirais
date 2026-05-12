# SPDX-License-Identifier: GPL-2.0-only

#' Tangent-space dimension via empirical-score rank
#'
#' Score basis (x - mean(x)) and (x^2 - mean(x^2)); returns rank of
#' empirical Gram matrix at tolerance 1e-10.
#'
#' @param x Numeric vector.
#' @return Named list with estimate (rank), n, method.
#' @references Kosorok (2008), Ch 6.
#' @export
ksr13_kosorok_tangent_space <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  s1 <- x - mean(x)
  s2 <- x^2 - mean(x^2)
  S <- cbind(s1, s2)
  G <- crossprod(S) / n
  rank <- as.integer(qr(G, tol = 1e-10)$rank)
  list(
    estimate = rank,
    n        = n,
    method   = "Tangent-space dim via empirical-score rank"
  )
}

# CANONICAL TEST
# set.seed(0); ksr13_kosorok_tangent_space(rnorm(200))

#' @rdname ksr13_kosorok_tangent_space
#' @keywords internal
#' @export
kosorok_tangent_space <- ksr13_kosorok_tangent_space
