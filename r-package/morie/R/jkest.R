# SPDX-License-Identifier: MIT OR Apache-2.0
#' Jackknife bias and variance (Quenouille 1956, Tukey 1958)
#'
#' Leave-one-out jackknife: bias_jack = (n-1)*(mean(T_minus_i) - T_hat),
#' var_jack = (n-1)/n * sum((T_minus_i - mean(T_minus_i))^2),
#' T_jack = n*T_hat - (n-1)*mean(T_minus_i).
#'
#' @param x numeric vector.
#' @param statistic function returning a scalar; default \code{mean}.
#' @return Named list: estimate (T_jack), theta_hat, bias, var, se, n, method.
#' @references Efron & Tibshirani (1993), Ch. 11.
#' @keywords internal
jkest <- function(x, statistic = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2L) {
    return(list(estimate = NA_real_, bias = NA_real_, se = NA_real_,
                n = n, method = "Jackknife (n<2)"))
  }
  if (is.null(statistic)) statistic <- mean
  T_hat <- statistic(x)
  T_loo <- vapply(seq_len(n), function(i) statistic(x[-i]), numeric(1))
  T_bar <- mean(T_loo)
  bias  <- (n - 1) * (T_bar - T_hat)
  T_jack <- n * T_hat - (n - 1) * T_bar
  var_jack <- (n - 1) / n * sum((T_loo - T_bar)^2)
  se <- sqrt(var_jack)
  list(estimate = as.numeric(T_jack),
       theta_hat = as.numeric(T_hat),
       bias = as.numeric(bias),
       var = as.numeric(var_jack),
       se = as.numeric(se),
       n = as.integer(n),
       method = "Jackknife (Quenouille 1956)")
}

# CANONICAL TEST
# r <- jkest(c(3, 5, 7, 9, 11))
# stopifnot(abs(r$theta_hat - 7) < 1e-12, abs(r$bias) < 1e-12)

#' @rdname jkest
#' @keywords internal
#' @export
jackknife_estimator <- jkest
