# SPDX-License-Identifier: GPL-2.0-only

#' Monte-Carlo power of the Wilcoxon signed-rank test
#' (Gibbons Ch 5.7.3)
#'
#' Simulates samples of size `length(x)` from Normal(effect_size, 1)
#' and reports the rejection rate of two-sided wilcox.test at level
#' alpha.
#'
#' @param x Numeric vector (only `length(x)` is used).
#' @param effect_size Location shift under H1.
#' @param alpha Test level.
#' @param nsim Replicates.
#' @param seed Reproducibility seed (NULL = no fix).
#' @return Named list: statistic (power), n, effect_size, alpha, nsim, se.
#' @importFrom stats wilcox.test rnorm
#' @export
wilcoxon_power <- function(x, effect_size = 0.5, alpha = 0.05,
                            nsim = 2000, seed = 0) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5)
    return(list(statistic = NA_real_, n = n, effect_size = effect_size,
                alpha = alpha, nsim = nsim, se = NA_real_,
                method = "Wilcoxon signed-rank power (Monte Carlo)"))
  if (!is.null(seed)) set.seed(seed)
  rejections <- 0L
  for (i in seq_len(nsim)) {
    s <- stats::rnorm(n, mean = effect_size, sd = 1)
    p <- tryCatch(
      suppressWarnings(stats::wilcox.test(s, exact = FALSE,
                                           correct = FALSE)$p.value),
      error = function(e) 1)
    if (!is.na(p) && p < alpha) rejections <- rejections + 1L
  }
  power <- rejections / nsim
  se <- sqrt(power * (1 - power) / nsim)
  list(
    statistic = power,
    n = n,
    effect_size = effect_size,
    alpha = alpha,
    nsim = nsim,
    se = se,
    method = "Wilcoxon signed-rank power (Monte Carlo)"
  )
}
