# SPDX-License-Identifier: MIT OR Apache-2.0
#' Copula parameter estimation (Gaussian/Clayton/Gumbel; Nelsen 2006)
#'
#' Kendall's tau inversion:
#'   Gaussian: rho   = sin(pi tau / 2)
#'   Clayton:  theta = 2 tau / (1 - tau)
#'   Gumbel:   theta = 1 / (1 - tau)
#'
#' @param x,y numeric marginal samples.
#' @param family "gaussian", "clayton", or "gumbel".
#' @return list: estimate, kendall_tau, se_tau, u, v, family, n, method.
#' @importFrom stats cor.test rank
#' @keywords internal
copul <- function(x, y, family = c("gaussian", "clayton", "gumbel")) {
  family <- match.arg(family)
  x <- as.numeric(x); y <- as.numeric(y); n <- min(length(x), length(y))
  if (n < 3L)
    return(list(estimate = NA_real_, n = n,
                method = paste0("copula-", family, " (n<3)")))
  tau <- stats::cor(x[seq_len(n)], y[seq_len(n)], method = "kendall")
  theta <- switch(family,
                  gaussian = sin(pi * tau / 2),
                  clayton  = if (tau < 1) 2 * tau / (1 - tau) else Inf,
                  gumbel   = if (tau < 1) 1 / (1 - tau) else Inf)
  u <- (rank(x[seq_len(n)]) - 0.5) / n
  v <- (rank(y[seq_len(n)]) - 0.5) / n
  list(estimate = as.numeric(theta),
       kendall_tau = as.numeric(tau),
       se_tau = sqrt((1 - tau^2) / n),
       u = u, v = v, family = family, n = as.integer(n),
       method = paste0("Copula ", family, " (rank-based; Nelsen 2006)"))
}

# CANONICAL TEST
# set.seed(0); Sigma <- matrix(c(1, 0.7, 0.7, 1), 2)
# z <- MASS::mvrnorm(500, c(0,0), Sigma)
# r <- copul(z[,1], z[,2], family = "gaussian")
# stopifnot(abs(r$estimate - 0.7) < 0.1)

#' @rdname copul
#' @keywords internal
#' @export
copula_estimation <- copul
