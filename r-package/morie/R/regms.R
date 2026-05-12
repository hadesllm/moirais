# SPDX-License-Identifier: GPL-2.0-only

#' Markov-switching regression (Hamilton 1989)
#'
#' Fit a constant-mean, switching-variance K-regime Markov-switching
#' model by EM (Hamilton filter).
#'
#' @param x Numeric univariate series.
#' @param k_regimes Number of latent regimes. Default 2.
#' @return Named list with \code{mu, sigma, transition,
#'   smoothed_probabilities, loglik, n, k_regimes, method}.
#' @export
regime_switching <- function(x, k_regimes = 2) {
  y <- as.numeric(x); n <- length(y)
  if (n < 4 * k_regimes) stop("Series too short for K regimes.")
  if (requireNamespace("MSwM", quietly = TRUE)) {
    df <- data.frame(y = y)
    base_fit <- lm(y ~ 1, data = df)
    msfit <- MSwM::msmFit(base_fit, k = k_regimes, sw = c(TRUE, TRUE))
    return(list(mu = as.numeric(msfit@Coef[, 1]),
                sigma = as.numeric(msfit@std),
                transition = msfit@transMat,
                smoothed_probabilities = msfit@Fit@smoProb,
                loglik = msfit@Fit@logLikel,
                n = n, k_regimes = k_regimes,
                method = sprintf("MSwM (K=%d)", k_regimes)))
  }
  mu <- seq(min(y), max(y), length.out = k_regimes)
  sig <- rep(max(sd(y), 1e-6), k_regimes)
  P <- matrix(1 / k_regimes, k_regimes, k_regimes)
  pi <- rep(1 / k_regimes, k_regimes)
  ll_prev <- -Inf
  for (it in seq_len(200)) {
    emit <- t(sapply(y,
      function(yt) dnorm(yt, mean = mu, sd = sig)))
    emit <- pmax(emit, 1e-300)
    alpha <- matrix(0, n, k_regimes); cv <- numeric(n)
    alpha[1, ] <- pi * emit[1, ]; cv[1] <- sum(alpha[1, ])
    alpha[1, ] <- alpha[1, ] / cv[1]
    for (t in 2:n) {
      alpha[t, ] <- (alpha[t - 1, ] %*% P) * emit[t, ]
      cv[t] <- sum(alpha[t, ]); alpha[t, ] <- alpha[t, ] / max(cv[t], 1e-300)
    }
    beta <- matrix(0, n, k_regimes); beta[n, ] <- 1
    for (t in (n - 1):1) {
      beta[t, ] <- P %*% (emit[t + 1, ] * beta[t + 1, ])
      beta[t, ] <- beta[t, ] / max(sum(beta[t, ]), 1e-300)
    }
    gamma <- alpha * beta
    gamma <- gamma / rowSums(gamma)
    xi <- array(0, c(n - 1, k_regimes, k_regimes))
    for (t in seq_len(n - 1)) {
      xi[t, , ] <- (alpha[t, ] %*% t(beta[t + 1, ] * emit[t + 1, ])) * P
      xi[t, , ] <- xi[t, , ] / max(sum(xi[t, , ]), 1e-300)
    }
    pi <- gamma[1, ]
    P <- apply(xi, c(2, 3), sum) /
      pmax(colSums(gamma[seq_len(n - 1), , drop = FALSE]), 1e-12)
    for (k in seq_len(k_regimes)) {
      wk <- gamma[, k]
      mu[k] <- sum(wk * y) / max(sum(wk), 1e-12)
      sig[k] <- max(sqrt(sum(wk * (y - mu[k])^2) / max(sum(wk), 1e-12)), 1e-6)
    }
    ll <- sum(log(pmax(cv, 1e-300)))
    if (abs(ll - ll_prev) < 1e-6) break
    ll_prev <- ll
  }
  list(mu = mu, sigma = sig, transition = P,
       smoothed_probabilities = gamma,
       loglik = ll_prev, n = n, k_regimes = k_regimes,
       method = sprintf("Markov switching via EM/Hamilton filter (K=%d, base R)",
                        k_regimes))
}
