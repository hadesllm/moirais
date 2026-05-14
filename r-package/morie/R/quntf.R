# SPDX-License-Identifier: MIT OR Apache-2.0
#' Nonparametric quantile function with asymptotic SEs (Parzen 1979)
#'
#' Q(tau) = inf over x of F_n(x) at least tau; asymptotic SE
#'   se(Q(tau)) ~~ sqrt(tau(1-tau) / (n f(Q(tau))^2)),
#' with f estimated by a Gaussian kernel (Silverman's rule bandwidth).
#'
#' @param x numeric vector.
#' @param taus probability levels in (0, 1).  Default c(0.1, 0.25, 0.5, 0.75, 0.9).
#' @return list: taus, quantiles, se, bandwidth, n, method.
#' @keywords internal
quntf <- function(x, taus = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2L)
    return(list(estimate = NA_real_, n = n, method = "Quantile fn (n<2)"))
  if (is.null(taus)) taus <- c(0.10, 0.25, 0.50, 0.75, 0.90)
  q <- stats::quantile(x, probs = taus, names = FALSE, type = 7)
  sd_x <- stats::sd(x)
  iqr <- diff(stats::quantile(x, c(0.25, 0.75), names = FALSE))
  h <- if (iqr > 0) 1.06 * min(sd_x, iqr / 1.34) * n^(-0.2)
       else 1.06 * sd_x * n^(-0.2)
  if (!is.finite(h) || h <= 0) h <- if (sd_x > 0) sd_x else 1
  fhat <- vapply(q, function(qi) mean(stats::dnorm(x, mean = qi, sd = h)),
                 numeric(1))
  se <- sqrt(taus * (1 - taus) / (n * fhat^2))
  list(taus = taus, quantiles = as.numeric(q), se = as.numeric(se),
       bandwidth = as.numeric(h),
       estimate = as.numeric(q[ceiling(length(q) / 2)]),
       n = as.integer(n),
       method = "Empirical quantile function (Parzen 1979)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(2000)
# r <- quntf(x, taus = 0.5)
# stopifnot(abs(r$quantiles[1]) < 0.1)

#' @rdname quntf
#' @keywords internal
#' @export
quantile_function <- quntf
