# SPDX-License-Identifier: MIT OR Apache-2.0

#' Synchronised longitudinal-panel simulation (R parity)
#'
#' Clean-room R parity of `morie.longitudinal_sim` for synchronised
#' multivariate longitudinal-panel simulation.  Implements SyncRNG,
#' VAR coefficient generation with stationarity preservation, MVN
#' draws under structured covariance kernels, and tidy panel output.
#'
#' Clean-room note: this module re-implements the techniques used in
#' the Hlozek--Bangari Collaborative-CIFAR-Catalyst project
#' (\url{https://github.com/bangari-19/Collaborative-CIFAR-Project-})
#' without copying any source.  That repository is unlicensed.  The
#' techniques themselves --- synchronised PRNG streams, lagged AR
#' coefficient matrices, multivariate normal generation under
#' Toeplitz / compound-symmetric covariance --- are standard methods
#' from Hamilton (1994) and Diggle, Liang, Zeger (1994), implemented
#' here independently.
#'
#' @name longitudinal_sim
NULL


#' Synchronised RNG seeded reproducibly for cross-language workflows
#'
#' Returns a function that mimics \code{stats::runif} but is seeded
#' from \code{seed}.  Pairs with \code{morie.longitudinal_sim.sync_rng}
#' on the Python side so the two emit identical streams when given the
#' same seed.
#'
#' @param seed Non-negative integer seed.
#' @return An environment with \code{rnorm}, \code{runif}, \code{sample}
#'   methods that share the same underlying RNG state.
#' @export
morie_sync_rng <- function(seed) {
  stopifnot(is.numeric(seed), length(seed) == 1L, seed >= 0,
            seed == as.integer(seed))
  set.seed(as.integer(seed), kind = "L'Ecuyer-CMRG")
  env <- new.env(parent = emptyenv())
  env$rnorm  <- function(n, mean = 0, sd = 1) stats::rnorm(n, mean, sd)
  env$runif  <- function(n, min = 0, max = 1) stats::runif(n, min, max)
  env$sample <- function(x, size, replace = FALSE) base::sample(x, size, replace)
  env
}


#' Generate a stationarity-preserving AR coefficient matrix
#'
#' @param p Dimension (number of variables).
#' @param rng An environment from \code{morie_sync_rng()}.
#' @param spectral_radius Target spectral radius < 1.
#' @param diagonal_bias Mixture weight between diagonal autoregression
#'   (1) and full off-diagonal coupling (0).
#' @return A p x p numeric matrix A.
#' @export
morie_generate_ar_coefficients <- function(p, rng,
                                            spectral_radius = 0.8,
                                            diagonal_bias = 0.4) {
  stopifnot(p >= 1, spectral_radius > 0, spectral_radius < 1)
  A_diag <- diagonal_bias * diag(p)
  A_off  <- (1 - diagonal_bias) * matrix(rng$rnorm(p * p) * 0.3, p, p)
  diag(A_off) <- 0
  A <- A_diag + A_off
  rho <- max(Mod(eigen(A, only.values = TRUE)$values))
  if (rho > 0) A <- A * (spectral_radius / rho)
  A
}


#' Generate a VAR(L) coefficient array as a 3-d list
#'
#' @param p Number of variables.
#' @param lags Number of lag matrices.
#' @param rng \code{morie_sync_rng()} environment.
#' @param spectral_radius Per-lag target spectral radius.
#' @param decay Geometric decay rate of spectral radius across lags.
#' @return A list of length \code{lags}, each a p x p matrix.
#' @export
morie_generate_var_coefficients <- function(p, lags, rng,
                                             spectral_radius = 0.8,
                                             decay = 0.6) {
  stopifnot(lags >= 1)
  lapply(seq_len(lags) - 1L, function(l) {
    morie_generate_ar_coefficients(p, rng,
                                    spectral_radius = spectral_radius * decay^l)
  })
}


#' Draw multivariate normal samples under a structured covariance
#'
#' @param n Number of samples.
#' @param p Dimension.
#' @param rng \code{morie_sync_rng()} environment.
#' @param kernel One of \code{"independent"}, \code{"ar1"},
#'   \code{"compound"}, \code{"toeplitz"}.
#' @param rho Correlation parameter.
#' @param mean Optional length-p mean vector.
#' @return An n x p matrix of samples.
#' @export
morie_mvn_with_covariance <- function(n, p, rng,
                                       kernel = c("ar1", "independent", "compound", "toeplitz"),
                                       rho = 0.5, mean = NULL) {
  kernel <- match.arg(kernel)
  if (is.null(mean)) mean <- rep(0, p)
  sigma <- switch(kernel,
    independent = diag(p),
    ar1         = ,
    toeplitz    = rho ^ abs(outer(seq_len(p), seq_len(p), "-")),
    compound    = (function() {
      m <- matrix(rho, p, p); diag(m) <- 1; m
    })()
  )
  L <- chol(sigma)
  z <- matrix(rng$rnorm(n * p), n, p)
  sweep(z %*% L, 2, mean, "+")
}


#' Simulate a longitudinal panel and return a tidy long-format
#' data.frame
#'
#' @param n_individuals Number of subjects.
#' @param n_timepoints Number of time-points per subject.
#' @param p_variables Number of variables.
#' @param cov_kernel Innovation covariance kernel.
#' @param cov_rho Correlation parameter.
#' @param ar_lags VAR lag order.
#' @param ar_spectral_radius Target spectral radius (per lag).
#' @param ar_decay Geometric decay across lags.
#' @param missing_fraction Probability of NA mask per entry.
#' @param outlier_fraction Probability of outlier amplification.
#' @param outlier_scale Multiplicative factor for outliers.
#' @param seed Non-negative integer seed.
#' @return A data.frame with columns subject_id, t, variable, value.
#' @export
#' @examples
#' if (FALSE) {
#'   df <- morie_simulate_longitudinal_panel(
#'     n_individuals = 30, n_timepoints = 10, p_variables = 4
#'   )
#'   head(df)
#' }
morie_simulate_longitudinal_panel <- function(
  n_individuals = 50, n_timepoints = 20, p_variables = 3,
  cov_kernel = "ar1", cov_rho = 0.5,
  ar_lags = 1L, ar_spectral_radius = 0.8, ar_decay = 0.6,
  missing_fraction = 0.0, outlier_fraction = 0.0, outlier_scale = 5.0,
  seed = 42L
) {
  rng <- morie_sync_rng(seed)
  A   <- morie_generate_var_coefficients(p_variables, ar_lags, rng,
                                          spectral_radius = ar_spectral_radius,
                                          decay = ar_decay)
  panel <- array(0, dim = c(n_individuals, n_timepoints, p_variables))
  for (i in seq_len(n_individuals)) {
    eps <- morie_mvn_with_covariance(n_timepoints, p_variables, rng,
                                      kernel = cov_kernel, rho = cov_rho)
    history <- vector("list", n_timepoints)
    for (t in seq_len(n_timepoints)) {
      x_new <- eps[t, ]
      for (l in seq_len(ar_lags)) {
        idx <- t - l
        if (idx >= 1) x_new <- x_new + drop(A[[l]] %*% history[[idx]])
      }
      history[[t]] <- x_new
      panel[i, t, ] <- x_new
    }
  }
  if (missing_fraction > 0) {
    mask <- array(rng$runif(prod(dim(panel))) < missing_fraction, dim(panel))
    panel[mask] <- NA_real_
  }
  if (outlier_fraction > 0) {
    omask <- array(rng$runif(prod(dim(panel))) < outlier_fraction, dim(panel))
    panel[omask] <- panel[omask] * outlier_scale
  }
  out <- expand.grid(subject_id = seq_len(n_individuals) - 1L,
                     t = seq_len(n_timepoints) - 1L,
                     variable = seq_len(p_variables) - 1L,
                     KEEP.OUT.ATTRS = FALSE)
  out$value <- as.numeric(panel)
  out[order(out$subject_id, out$t, out$variable), ]
}
