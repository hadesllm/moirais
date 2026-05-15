# SPDX-License-Identifier: AGPL-3.0-or-later

#' Bayesian ideal-point estimation (Armstrong Ch 5)
#'
#' Metropolis-within-Gibbs surrogate for Clinton-Jackman-Rivers (2004)
#' with N(0,1) prior on x and N(0, 5) priors on (alpha, beta).
#'
#' @param x Binary vote matrix (n by m).
#' @param n_iter Number of MCMC sweeps (default 400).
#' @param burn Burn-in sweeps (default 100).
#' @param seed RNG seed.
#' @param deterministic_seed Optional integer; if supplied, RNG state is
#'   derived via [morie_det_rng()] keyed on ("bysid", deterministic_seed)
#'   so Py<->R streams agree on the canonical fixture.  When `NULL`
#'   (default) behaviour is unchanged.
#' @return Named list with `x_mean`, `x_sd`, `x_ci`, `alpha`, `beta`,
#'   `n_iter`, `method`.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
bysid <- function(x, n_iter = 400L, burn = 100L, seed = 0L,
                  deterministic_seed = NULL) {
  logistic <- function(z) 1 / (1 + exp(-pmin(pmax(z, -30), 30)))
  if (!is.null(deterministic_seed)) {
    morie::morie_det_rng("bysid", deterministic_seed)
  } else {
    set.seed(seed)
  }
  M <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1L)
  n <- nrow(M); m <- ncol(M)
  if (n < 2L)
    return(list(x_mean = rep(NA_real_, n), x_sd = rep(NA_real_, n),
                x_ci = matrix(NA_real_, n, 2L),
                alpha = rep(NA_real_, m), beta = rep(NA_real_, m),
                n_iter = 0L, method = "bayesian_ideal_points"))
  Mc <- M - matrix(colMeans(M, na.rm = TRUE), n, m, byrow = TRUE)
  Mc[is.na(Mc)] <- 0
  sv <- tryCatch(svd(Mc, nu = 1L, nv = 0L), error = function(e) NULL)
  x_cur <- if (!is.null(sv)) sv$u[, 1] * sv$d[1] else stats::rnorm(n)
  x_cur <- (x_cur - mean(x_cur)) / (stats::sd(x_cur) + 1e-12)
  a_cur <- rep(1, m); b_cur <- rep(0, m)
  step_x <- 0.4; step_ab <- 0.3
  loglik <- function(xv, av, bv) {
    Z <- sweep(outer(xv, bv, FUN = "-"), 2L, av, "*")
    P <- logistic(Z)
    mask <- !is.na(M)
    sum(ifelse(mask, M * log(P + 1e-12) + (1 - M) * log(1 - P + 1e-12), 0))
  }
  ll_cur <- loglik(x_cur, a_cur, b_cur)
  samples <- list(); a_samples <- list(); b_samples <- list()
  for (t in seq_len(n_iter)) {
    xp <- x_cur + step_x * stats::rnorm(n)
    llp <- loglik(xp, a_cur, b_cur)
    la <- (llp - 0.5 * sum(xp^2)) - (ll_cur - 0.5 * sum(x_cur^2))
    if (log(stats::runif(1)) < la) { x_cur <- xp; ll_cur <- llp }
    ap <- a_cur + step_ab * stats::rnorm(m)
    llp <- loglik(x_cur, ap, b_cur)
    la <- (llp - 0.5 * sum(ap^2) / 25) - (ll_cur - 0.5 * sum(a_cur^2) / 25)
    if (log(stats::runif(1)) < la) { a_cur <- ap; ll_cur <- llp }
    bp <- b_cur + step_ab * stats::rnorm(m)
    llp <- loglik(x_cur, a_cur, bp)
    la <- (llp - 0.5 * sum(bp^2) / 25) - (ll_cur - 0.5 * sum(b_cur^2) / 25)
    if (log(stats::runif(1)) < la) { b_cur <- bp; ll_cur <- llp }
    if (t > burn) {
      xs <- (x_cur - mean(x_cur)) / (stats::sd(x_cur) + 1e-12)
      samples[[length(samples) + 1L]] <- xs
      a_samples[[length(a_samples) + 1L]] <- a_cur
      b_samples[[length(b_samples) + 1L]] <- b_cur
    }
  }
  if (length(samples) == 0L) {
    return(list(x_mean = rep(NA_real_, n), x_sd = rep(NA_real_, n),
                x_ci = matrix(NA_real_, n, 2L),
                alpha = rep(NA_real_, m), beta = rep(NA_real_, m),
                n_iter = n_iter, method = "bayesian_ideal_points"))
  }
  arr <- do.call(rbind, samples)
  x_mean <- colMeans(arr); x_sd <- apply(arr, 2L, stats::sd)
  x_ci <- t(apply(arr, 2L,
                  function(z) stats::quantile(z, c(0.025, 0.975))))
  a_mean <- colMeans(do.call(rbind, a_samples))
  b_mean <- colMeans(do.call(rbind, b_samples))
  list(x_mean = x_mean, x_sd = x_sd, x_ci = x_ci,
       alpha = a_mean, beta = b_mean,
       n_iter = n_iter, method = "bayesian_ideal_points")
}

#' @keywords internal
#' @rdname bysid
#' @export
bayesian_ideal_points <- bysid
