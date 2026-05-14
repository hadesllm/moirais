# SPDX-License-Identifier: MIT OR Apache-2.0

#' DP mixture density estimate (Neal 2000 algorithm 3)
#' @param x numeric vector
#' @param alpha,sigma DP and within-cluster sd (sigma defaults to Silverman bw)
#' @param grid evaluation grid
#' @param n_iter,burn,seed Gibbs settings
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("ghdpm", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return named list with `estimate`, `grid`, `density`, `k_post`, `n`
#' @importFrom utils head tail
#' @export
ghosal_dpmixture_density <- function(x, alpha = 1.0, sigma = NULL,
                                       grid = NULL, n_iter = 120, burn = 40,
                                       seed = 0, deterministic_seed = NULL) {
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("ghdpm", deterministic_seed)
  } else {
    set.seed(seed)
  }
  x <- as.numeric(x); n <- length(x)
  if (n == 0) return(list(estimate = NA_real_, n = 0,
                          method = "DP-mixture density (empty input)"))
  if (is.null(sigma)) {
    s <- if (n > 1) sd(x) else 1
    sigma <- 1.06 * max(s, 1e-6) * n^(-1/5)
  }
  sigma <- max(sigma, 1e-6)
  m0 <- mean(x); s0 <- if (n > 1) sd(x) else 1
  s0 <- max(s0, 1e-3)
  if (is.null(grid)) grid <- seq(min(x) - s0, max(x) + s0, length.out = 51)
  labels <- rep(0L, n)
  k_chain <- integer(0); f_chain <- list()
  new_log <- function(xi) stats::dnorm(xi, mean = m0, sd = sqrt(sigma^2 + s0^2), log = TRUE)
  cluster_post <- function(xs) {
    nk <- length(xs); v <- 1 / (1/s0^2 + nk/sigma^2)
    m <- v * (m0/s0^2 + sum(xs)/sigma^2)
    list(m = m, v = v)
  }
  in_log <- function(xi, xs) {
    cp <- cluster_post(xs)
    stats::dnorm(xi, mean = cp$m, sd = sqrt(cp$v + sigma^2), log = TRUE)
  }
  for (it in seq_len(n_iter)) {
    for (i in seq_len(n)) {
      old <- labels[i]; labels[i] <- -1L
      uniq <- sort(unique(labels[labels >= 0]))
      lps <- numeric(length(uniq) + 1L)
      for (j in seq_along(uniq)) {
        xs <- x[labels == uniq[j]]
        lps[j] <- log(length(xs)) + in_log(x[i], xs)
      }
      lps[length(uniq) + 1L] <- log(alpha) + new_log(x[i])
      lps <- lps - max(lps); probs <- exp(lps); probs <- probs/sum(probs)
      choice <- sample.int(length(probs), 1, prob = probs)
      if (choice == length(uniq) + 1L) {
        labels[i] <- if (length(uniq)) max(uniq) + 1L else 0L
      } else {
        labels[i] <- uniq[choice]
      }
    }
    if (it > burn) {
      uniq <- sort(unique(labels))
      f <- numeric(length(grid))
      for (k in uniq) {
        xs <- x[labels == k]; cp <- cluster_post(xs)
        f <- f + (length(xs) / (alpha + n)) *
          stats::dnorm(grid, mean = cp$m, sd = sqrt(cp$v + sigma^2))
      }
      f <- f + (alpha / (alpha + n)) *
        stats::dnorm(grid, mean = m0, sd = sqrt(sigma^2 + s0^2))
      f_chain[[length(f_chain) + 1L]] <- f
      k_chain <- c(k_chain, length(uniq))
    }
  }
  density <- Reduce("+", f_chain) / length(f_chain)
  num <- sum(diff(grid) * (head(density * grid, -1) + tail(density * grid, -1))) / 2
  den <- sum(diff(grid) * (head(density, -1) + tail(density, -1))) / 2
  est <- num / max(den, 1e-12)
  list(estimate = est, grid = grid, density = density,
       k_post = mean(k_chain), n = n, alpha = alpha, sigma = sigma,
       method = "DP-mixture density via collapsed Gibbs (Neal 2000 Alg 3)")
}
