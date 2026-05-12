# SPDX-License-Identifier: GPL-2.0-only

#' Fauzi: KDFE bias and variance properties (Ch 2)
#'
#' Kernel distribution-function estimator (KDFE)
#' \eqn{\hat F_h(t) = n^{-1} \sum_i \Phi((t-X_i)/h)} and its asymptotic
#' bias and variance:
#' \deqn{\mathrm{Bias} = (h^2/2)\mu_2(K) f'(t),
#'       \quad \mathrm{Var}=F(t)(1-F(t))/n - 2 h r(K) f(t)/n.}
#' For the Gaussian kernel \eqn{\mu_2=1, r(K)=1/(2\sqrt\pi)}.
#'
#' @param x Numeric vector.
#' @param t Evaluation point; default = median(x).
#' @param h Bandwidth; default = Silverman's rule.
#' @return Named list: estimate, bias, variance, se, h, t, n, method.
#' @importFrom stats median sd quantile dnorm pnorm
#' @export
fzkdf <- function(x, t = NULL, h = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2L) return(list(estimate = NA_real_, n = n,
                           method = "fzkdf - too few obs"))
  if (is.null(t)) t <- stats::median(x)
  if (is.null(h)) h <- .morie_silverman_h(x)
  mu2 <- 1.0
  rK  <- 1 / (2 * sqrt(pi))
  z <- (t - x) / h
  F_hat <- mean(stats::pnorm(z))
  f_hat <- mean(stats::dnorm(z) / h)
  fp_hat <- mean(-z * stats::dnorm(z) / (h * h))
  bias <- (h^2 / 2) * mu2 * fp_hat
  var <- max(F_hat * (1 - F_hat) / n - 2 * h * rK * f_hat / n, 0)
  list(estimate = F_hat, bias = bias, variance = var,
       se = sqrt(var), h = h, t = t, n = n,
       method = "Fauzi KDFE bias-variance (Ch 2)")
}

.morie_silverman_h <- function(x) {
  n <- length(x); s <- stats::sd(x)
  iq <- diff(stats::quantile(x, c(.25, .75), names = FALSE)) / 1.34
  sigma <- if (iq > 0) min(s, iq) else s
  if (sigma <= 0) sigma <- 1
  1.06 * sigma * n^(-1/5)
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(500)
# r <- fzkdf(x, t = 0); stopifnot(abs(r$estimate - 0.5) < 0.1)
