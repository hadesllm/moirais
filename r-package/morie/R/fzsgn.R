# SPDX-License-Identifier: GPL-2.0-only

#' Fauzi: Smoothed sign test (Ch 5)
#'
#' \eqn{S_n=\sum_i \Phi((X_i-\theta_0)/h)},
#' z = (S_n - n/2)/sqrt(n/4) ~ N(0,1).
#'
#' @param x Numeric vector.
#' @param theta0 Null median; default 0.
#' @param h Bandwidth; default = Silverman.
#' @param alternative "two-sided", "greater", "less".
#' @return Named list with statistic, z, p_value, theta0, h, n, method.
#' @importFrom stats pnorm
#' @export
fzsgn <- function(x, theta0 = 0, h = NULL, alternative = "two-sided") {
  x <- as.numeric(x); n <- length(x)
  if (n < 5L) return(list(statistic = NA_real_, p_value = NA_real_, n = n,
                           method = "fzsgn - too few obs"))
  if (is.null(h)) h <- .morie_silverman_h(x)
  S_n <- sum(stats::pnorm((x - theta0) / h))
  z <- (S_n - n / 2) / sqrt(n / 4)
  p <- switch(alternative,
              "two-sided" = 2 * (1 - stats::pnorm(abs(z))),
              "greater"   = 1 - stats::pnorm(z),
              "less"      = stats::pnorm(z),
              stop("alternative must be two-sided/greater/less"))
  list(statistic = S_n, z = z, p_value = p,
       theta0 = theta0, h = h, n = n,
       method = sprintf("Fauzi smoothed sign test (%s) (Ch 5)", alternative))
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(500); r <- fzsgn(x, 0); stopifnot(r$p_value > 0.05)

#' @rdname fzsgn
#' @keywords internal
#' @export
fauzi_smoothed_sign <- fzsgn
